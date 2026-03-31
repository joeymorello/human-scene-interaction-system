"use client";

import { useEffect, useRef } from "react";
import type { InteractionEvent } from "@/app/page";

type EventLogProps = {
  events: InteractionEvent[];
  currentTime: number;
};

export default function EventLog({ events, currentTime }: EventLogProps) {
  const activeRef = useRef<HTMLLIElement>(null);

  // Auto-scroll to active event
  useEffect(() => {
    activeRef.current?.scrollIntoView({ behavior: "smooth", block: "center" });
  }, [currentTime]);

  if (events.length === 0) {
    return (
      <div className="flex h-full items-center justify-center text-sm text-gray-600">
        No interaction events detected
      </div>
    );
  }

  return (
    <ul className="h-full overflow-y-auto p-3 font-mono text-sm">
      {events.map((evt, i) => {
        const isActive =
          Math.abs(currentTime - evt.time) < 0.5 ||
          (i < events.length - 1 &&
            currentTime >= evt.time &&
            currentTime < events[i + 1].time);

        return (
          <li
            key={i}
            ref={isActive ? activeRef : null}
            className={`rounded px-2 py-1.5 transition-colors ${
              isActive
                ? "bg-blue-600/20 text-blue-300"
                : "text-gray-400 hover:bg-gray-800"
            }`}
          >
            <span className="text-gray-600">[{evt.time.toFixed(1)}s]</span>{" "}
            <span className="text-yellow-400">{evt.action}</span>{" "}
            <span className="font-semibold text-gray-200">{evt.object}</span>
            <span className="ml-2 text-xs text-gray-600">
              {(evt.confidence * 100).toFixed(0)}%
            </span>
          </li>
        );
      })}
    </ul>
  );
}
