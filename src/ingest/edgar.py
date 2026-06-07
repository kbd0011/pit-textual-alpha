"""Rate-limited, point-in-time-correct SEC EDGAR client.

The crown jewel of this project: every filing is tagged with its `acceptanceDateTime` and a derived
`dissemination_time`, so downstream code can gate on *what was actually knowable* at a given moment.

SEC rules respected here:
- A descriptive `User-Agent` header is required.
- No more than 10 requests/second (we self-throttle below that).
- Filings accepted after 17:30 ET (22:00 ET for ownership forms) disseminate the next business day.

This module is intentionally dependency-light (requests + stdlib + pandas) so unit tests can mock the network.
"""
from __future__ import annotations

import argparse
import re
import time
from collections.abc import Iterable, Iterator
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from datetime import time as dtime
from zoneinfo import ZoneInfo

import requests

ET = ZoneInfo("America/New_York")
SUBMISSIONS_URL = "https://data.sec.gov/submissions/CIK{cik:010d}.json"
ARCHIVES_URL = "https://www.sec.gov/Archives/edgar/data/{cik}/{accession_nodash}/{doc}"

AFTER_HOURS_CUTOFF = dtime(17, 30)
OWNERSHIP_CUTOFF = dtime(22, 0)
OWNERSHIP_FORMS = {"3", "4", "5", "3/A", "4/A", "5/A"}


@dataclass
class Filing:
    """A single EDGAR filing with point-in-time metadata."""

    cik: int
    form: str
    accession: str
    filing_date: str                 # date only — DO NOT use for gating
    report_date: str | None          # period covered — DO NOT use for gating
    acceptance_datetime: datetime    # the dissemination timestamp basis (ET)
    primary_document: str
    dissemination_time: datetime = field(init=False)

    def __post_init__(self) -> None:
        self.dissemination_time = _dissemination_time(self.acceptance_datetime, self.form)

    @property
    def primary_doc_url(self) -> str:
        nodash = self.accession.replace("-", "")
        return ARCHIVES_URL.format(cik=self.cik, accession_nodash=nodash, doc=self.primary_document)


def _next_business_day_open(dt: datetime) -> datetime:
    """Roll a datetime forward to the next business day at 09:30 ET (ignores market holidays;
    swap in pandas-market-calendars for production)."""
    nxt = (dt + timedelta(days=1)).date()
    while nxt.weekday() >= 5:  # Sat/Sun
        nxt = nxt + timedelta(days=1)
    return datetime.combine(nxt, dtime(9, 30), tzinfo=ET)


def _dissemination_time(acceptance: datetime, form: str) -> datetime:
    """When the market could first act on a filing.

    If accepted at/after the cutoff, it disseminates the next business day; otherwise same instant.
    """
    if acceptance.tzinfo is None:
        acceptance = acceptance.replace(tzinfo=ET)
    cutoff = OWNERSHIP_CUTOFF if form in OWNERSHIP_FORMS else AFTER_HOURS_CUTOFF
    if acceptance.timetz().replace(tzinfo=None) >= cutoff:
        return _next_business_day_open(acceptance)
    return acceptance


class EdgarClient:
    """Minimal throttled EDGAR client."""

    def __init__(self, user_agent: str, max_rps: float = 8.0, session: requests.Session | None = None):
        if not user_agent or "@" not in user_agent:
            raise ValueError("SEC requires a descriptive User-Agent like 'Name email@domain.com'")
        self.user_agent = user_agent
        self._min_interval = 1.0 / max_rps
        self._last = 0.0
        self.session = session or requests.Session()
        self.session.headers.update({"User-Agent": user_agent, "Accept-Encoding": "gzip, deflate"})

    def _throttle(self) -> None:
        elapsed = time.monotonic() - self._last
        if elapsed < self._min_interval:
            time.sleep(self._min_interval - elapsed)
        self._last = time.monotonic()

    def _get(self, url: str) -> requests.Response:
        self._throttle()
        resp = self.session.get(url, timeout=30)
        resp.raise_for_status()
        return resp

    def get_submissions(self, cik: int) -> dict:
        """Fetch the submissions JSON for a CIK."""
        return self._get(SUBMISSIONS_URL.format(cik=cik)).json()

    def iter_filings(self, cik: int, forms: Iterable[str] | None = None) -> Iterator[Filing]:
        """Yield Filing objects (recent set) for a CIK, optionally filtered by form type."""
        data = self.get_submissions(cik)
        recent = data["filings"]["recent"]
        wanted = set(forms) if forms else None
        n = len(recent["accessionNumber"])
        for i in range(n):
            form = recent["form"][i]
            if wanted and form not in wanted:
                continue
            acc_raw = recent["acceptanceDateTime"][i]  # e.g. "2023-11-03T18:01:23.000Z" or with offset
            acceptance = _parse_acceptance(acc_raw)
            yield Filing(
                cik=cik,
                form=form,
                accession=recent["accessionNumber"][i],
                filing_date=recent["filingDate"][i],
                report_date=recent["reportDate"][i] or None,
                acceptance_datetime=acceptance,
                primary_document=recent["primaryDocument"][i],
            )

    def fetch_document(self, filing: Filing) -> str:
        """Download a filing's primary document (HTML/text)."""
        return self._get(filing.primary_doc_url).text


def _parse_acceptance(raw: str) -> datetime:
    """Parse EDGAR acceptanceDateTime into an ET-aware datetime."""
    raw = raw.replace("Z", "+00:00")
    dt = datetime.fromisoformat(raw)
    if dt.tzinfo is None:
        # EDGAR acceptance times are Eastern; localize.
        return dt.replace(tzinfo=ET)
    return dt.astimezone(ET)


def knowable_filings(filings: list[Filing], as_of: datetime) -> list[Filing]:
    """Return only filings whose dissemination_time <= as_of (the point-in-time guard)."""
    if as_of.tzinfo is None:
        as_of = as_of.replace(tzinfo=ET)
    return [f for f in filings if f.dissemination_time <= as_of]


# Section boundary patterns, matched against whitespace-normalized text (case-insensitive).
# MD&A is Item 7 in a 10-K and Item 2 in a 10-Q; we try both so one function handles either form.
_MDNA_STARTS = (
    r"item\s*7\b[\s\.\:\-–—]{0,8}management.{0,3}s\s+discussion\s+and\s+analysis",
    r"item\s*2\b[\s\.\:\-–—]{0,8}management.{0,3}s\s+discussion\s+and\s+analysis",
)
_MDNA_ENDS = (
    r"item\s*7a\b",                                   # 10-K: Quantitative and Qualitative Disclosures
    r"item\s*8\b",                                    # 10-K: Financial Statements
    r"item\s*3\b[\s\.\:\-–—]{0,8}quantitative",  # 10-Q: Quantitative and Qualitative
    r"item\s*4\b[\s\.\:\-–—]{0,8}controls",      # 10-Q: Controls and Procedures
)
_RISK_STARTS = (r"item\s*1a\b[\s\.\:\-–—]{0,8}risk\s+factors",)
_RISK_ENDS = (
    r"item\s*1b\b",                                   # Unresolved Staff Comments
    r"item\s*2\b[\s\.\:\-–—]{0,8}properties",
    r"item\s*1c\b",                                   # Cybersecurity (newer filings)
)

_MIN_SECTION_CHARS = 400  # below this we assume we matched a table-of-contents entry, not the body


def _html_to_text(html: str) -> str:
    """HTML -> whitespace-normalized plain text, dropping scripts, styles, tables and XBRL headers.

    Tables are removed because financial tables are numeric noise for sentiment and their digits would
    otherwise contaminate the text. Whitespace (incl. non-breaking spaces) is collapsed to single spaces.
    """
    from bs4 import BeautifulSoup

    soup = BeautifulSoup(html, "lxml")
    for tag in soup(["script", "style", "table"]):
        tag.decompose()
    for tag in soup.find_all(re.compile(r"^ix:", re.IGNORECASE)):  # inline-XBRL wrappers
        tag.decompose()
    text = soup.get_text(separator=" ")
    text = text.replace("\xa0", " ")
    return re.sub(r"\s+", " ", text).strip()


def _longest_section(text: str, start_pats: Iterable[str], end_pats: Iterable[str]) -> str | None:
    """Return the longest text slice from any start marker to the next end marker after it.

    A 10-K names every section twice — once in the table of contents, once as the real body. The TOC entry is
    immediately followed by the next TOC entry, so its start->end slice is tiny; the body slice is long. Taking
    the longest candidate robustly selects the body over the TOC.
    """
    starts = [m.start() for p in start_pats for m in re.finditer(p, text, re.IGNORECASE)]
    ends = [m.start() for p in end_pats for m in re.finditer(p, text, re.IGNORECASE)]
    if not starts:
        return None
    ends.sort()
    best: str | None = None
    for s in starts:
        nxt = next((e for e in ends if e > s), None)
        candidate = text[s:nxt] if nxt is not None else text[s:]
        if best is None or len(candidate) > len(best):
            best = candidate
    if best is not None and len(best) >= _MIN_SECTION_CHARS:
        return best.strip()
    return None


def extract_mdna(html: str) -> str:
    """Extract Management's Discussion & Analysis from a 10-K (Item 7) or 10-Q (Item 2).

    Strips tags/tables/exhibits, locates the MD&A body between its header and the next item (7A/8 for a 10-K,
    3/4 for a 10-Q), and normalizes whitespace. Falls back to Item 1A risk factors when MD&A is absent.

    Parameters
    ----------
    html : str
        Raw primary-document HTML (or plain text) of the filing.

    Returns
    -------
    str
        The cleaned MD&A text, or the risk-factors text if MD&A is not found.

    Raises
    ------
    ValueError
        If neither an MD&A nor a risk-factors section can be located.
    """
    text = _html_to_text(html)
    mdna = _longest_section(text, _MDNA_STARTS, _MDNA_ENDS)
    if mdna is not None:
        return mdna
    risk = _longest_section(text, _RISK_STARTS, _RISK_ENDS)
    if risk is not None:
        return risk
    raise ValueError("Could not locate an MD&A (Item 7/2) or risk-factors (Item 1A) section")


def _cli() -> None:
    p = argparse.ArgumentParser(description="EDGAR point-in-time filing lister")
    p.add_argument("--cik", type=int, required=True)
    p.add_argument("--forms", nargs="*", default=["10-K"])
    p.add_argument("--user-agent", default=None)
    args = p.parse_args()
    import os

    ua = args.user_agent or os.environ.get("SEC_USER_AGENT", "")
    client = EdgarClient(user_agent=ua)
    for f in client.iter_filings(args.cik, forms=args.forms):
        print(f"{f.form:6s} acc={f.acceptance_datetime.isoformat()} "
              f"disseminated={f.dissemination_time.isoformat()} {f.accession}")


if __name__ == "__main__":
    _cli()
