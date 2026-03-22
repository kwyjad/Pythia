import type { ReactNode } from "react";

type CollapsiblePanelProps = {
  title: string;
  children: ReactNode;
  defaultOpen?: boolean;
  onExpand?: () => void;
};

const CollapsiblePanel = ({
  title,
  children,
  defaultOpen = false,
  onExpand,
}: CollapsiblePanelProps) => {
  return (
    <details
      className="rounded-lg border border-fred-secondary bg-fred-surface"
      open={defaultOpen}
      onToggle={(event) => {
        if ((event.target as HTMLDetailsElement).open && onExpand) {
          onExpand();
        }
      }}
    >
      <summary className="cursor-pointer select-none px-4 py-2 text-sm font-semibold text-fred-secondary">
        {title}
      </summary>
      <div className="border-t border-fred-secondary px-4 py-3">{children}</div>
    </details>
  );
};

export default CollapsiblePanel;
