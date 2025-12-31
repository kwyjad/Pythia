import Link from "next/link";

const Nav = () => {
  return (
    <nav className="border-b border-fred-border bg-fred-surface">
      <div className="flex w-full items-center justify-between px-6 py-4">
        <Link href="/" className="flex items-center gap-3">
          <img
            src="/brand/fred_logo.png"
            alt="Fred logo"
            className="h-8 w-8"
          />
          <div className="flex flex-col">
            <span className="text-lg font-semibold text-fred-primary">
              Fred: Humanitarian Forecasting System
            </span>
            <span className="text-xs text-fred-muted">
              An AI-driven end-to-end humanitarian impact forecasting system
            </span>
          </div>
        </Link>
        <div className="flex items-center gap-4 text-sm">
          <Link className="text-fred-text/80 hover:text-fred-primary" href="/">
            Risk Index
          </Link>
          <Link className="text-fred-text/80 hover:text-fred-primary" href="/countries">
            Countries
          </Link>
          <Link className="text-fred-text/80 hover:text-fred-primary" href="/questions">
            Forecasts
          </Link>
          <Link className="text-fred-text/80 hover:text-fred-primary" href="/costs">
            Costs
          </Link>
          <Link className="text-fred-text/80 hover:text-fred-primary" href="/downloads">
            Downloads
          </Link>
          <Link className="text-fred-text/80 hover:text-fred-primary" href="/about">
            About
          </Link>
        </div>
      </div>
    </nav>
  );
};

export default Nav;
