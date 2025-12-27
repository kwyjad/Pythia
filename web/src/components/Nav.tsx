import Link from "next/link";

const Nav = () => {
  return (
    <nav className="border-b border-slate-800 bg-slate-950">
      <div className="mx-auto flex w-full max-w-6xl items-center justify-between px-6 py-4">
        <Link href="/" className="text-lg font-semibold text-white">
          Pythia Dashboard
        </Link>
        <div className="flex items-center gap-4 text-sm text-slate-300">
          <Link href="/">Overview</Link>
          <Link href="/countries/USA">Country</Link>
          <Link href="/questions/sample">Question</Link>
        </div>
      </div>
    </nav>
  );
};

export default Nav;
