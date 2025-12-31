import Link from "next/link";

const Nav = () => {
  return (
    <nav className="border-b border-fredBorder bg-fredSurface">
      <div className="flex w-full items-center justify-between px-6 py-4">
        <Link href="/" className="flex items-center gap-3">
          <img
            src="/brand/fred_logo.png"
            alt="Fred logo"
            className="h-8 w-8"
          />
          <div className="flex flex-col">
            <span className="text-lg font-semibold text-fredPrimary">
              Fred: Humanitarian Forecasting System
            </span>
            <span className="text-xs text-fredMuted">
              An AI-driven end-to-end humanitarian impact forecasting system
            </span>
          </div>
        </Link>
        <div className="flex items-center gap-4 text-sm">
          <Link className="text-fredText/80 hover:text-fredPrimary" href="/">
            Risk Index
          </Link>
          <Link className="text-fredText/80 hover:text-fredPrimary" href="/countries">
            Countries
          </Link>
          <Link className="text-fredText/80 hover:text-fredPrimary" href="/questions">
            Forecasts
          </Link>
          <Link className="text-fredText/80 hover:text-fredPrimary" href="/costs">
            Costs
          </Link>
          <Link className="text-fredText/80 hover:text-fredPrimary" href="/downloads">
            Downloads
          </Link>
          <Link className="text-fredText/80 hover:text-fredPrimary" href="/about">
            About
          </Link>
        </div>
      </div>
    </nav>
  );
};

export default Nav;
