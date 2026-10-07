// Training sample for the arko_index PGO profiles: a React list component.
// Written for this purpose; it is parsed, never run.
import React, { useEffect, useMemo, useState } from "react";

interface Job {
  id: string;
  title: string;
  state: "queued" | "running" | "done" | "failed";
  progress?: number;
}

type Props = {
  jobs: Job[];
  onRetry?: (id: string) => void;
  filter?: string;
};

function Badge({ state }: { state: Job["state"] }) {
  const color = state === "failed" ? "red" : state === "done" ? "green" : "gray";
  return <span className={`badge badge-${color}`}>{state}</span>;
}

export const JobList: React.FC<Props> = ({ jobs, onRetry, filter = "" }) => {
  const [expanded, setExpanded] = useState<Record<string, boolean>>({});
  const visible = useMemo(
    () => jobs.filter((job) => job.title.toLowerCase().includes(filter.toLowerCase())),
    [jobs, filter],
  );

  useEffect(() => {
    if (visible.length === 0) {
      document.title = "No jobs";
    }
    return () => {
      document.title = "Jobs";
    };
  }, [visible.length]);

  return (
    <ul className="job-list" data-count={visible.length}>
      {visible.map((job) => (
        <li key={job.id} onClick={() => setExpanded({ ...expanded, [job.id]: !expanded[job.id] })}>
          <strong>{job.title}</strong> <Badge state={job.state} />
          {job.progress !== undefined && <progress value={job.progress} max={100} />}
          {expanded[job.id] ? (
            <div className="details">
              <code>{job.id}</code>
              {job.state === "failed" && onRetry && <button onClick={() => onRetry(job.id)}>Retry</button>}
            </div>
          ) : null}
        </li>
      ))}
      <>{visible.length > 10 ? <li className="more">and more&hellip;</li> : null}</>
    </ul>
  );
};

export default JobList;
