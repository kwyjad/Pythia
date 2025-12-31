import Link from "next/link";

const Nav = () => {
  return (
    <nav className="border-b border-fred-secondary bg-fred-surface">
      <div className="flex w-full items-center justify-between px-6 py-4">
        <Link href="/" className="flex items-center gap-3">
          <img src="/brand/fred_logo.png" alt="Fred logo" className="h-24 w-24" />
          <div className="flex flex-col">
            <span className="text-2xl font-semibold text-fred-primary">
              Fred: Humanitarian Forecasting System
            </span>
            <span className="text-xs text-fred-muted">
              An AI-driven end-to-end humanitarian impact forecasting system
            </span>
          </div>
        </Link>
        <div className="flex items-center gap-4 text-sm">
          <Link
            className="text-fred-primary font-semibold hover:text-fred-secondary"
            href="/"
          >
            Risk Index
          </Link>
          <Link
            className="text-fred-primary font-semibold hover:text-fred-secondary"
            href="/countries"
          >
            Countries
          </Link>
          <Link
            className="text-fred-primary font-semibold hover:text-fred-secondary"
            href="/questions"
          >
            Forecasts
          </Link>
          <Link
            className="text-fred-primary font-semibold hover:text-fred-secondary"
            href="/costs"
          >
            Costs
          </Link>
          <Link
            className="text-fred-primary font-semibold hover:text-fred-secondary"
            href="/downloads"
          >
            Downloads
          </Link>
          <Link
            className="text-fred-primary font-semibold hover:text-fred-secondary"
            href="/about"
          >
            About
          </Link>
        </div>
      </div>
    </nav>
  );
};

export default Nav;
