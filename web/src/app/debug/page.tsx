import DebugClient from "./DebugClient";

export const dynamic = "force-dynamic";
export const revalidate = 0;

const DebugPage = () => {
  if (process.env.NODE_ENV !== "production") {
    console.log("[page] dynamic=force-dynamic", { route: "/debug" });
  }

  return (
    <div className="space-y-6">
      <section>
        <h1 className="text-3xl font-semibold">Debug</h1>
        <p className="text-sm text-fred-text">
          Review Horizon Scan triage diagnostics and downstream counts.
        </p>
      </section>
      <DebugClient />
    </div>
  );
};

export default DebugPage;
