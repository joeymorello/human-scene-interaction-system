"use client";

import { useState, useRef, useCallback } from "react";
import UploadPanel from "@/components/ui/UploadPanel";
import VideoPanel from "@/components/ui/VideoPanel";
import dynamic from "next/dynamic";
const SceneCanvas = dynamic(() => import("@/components/three/SceneCanvas"), {
  ssr: false,
  loading: () => (
    <div className="flex h-full items-center justify-center text-gray-500">
      Loading 3D viewer...
    </div>
  ),
});
import EventLog from "@/components/ui/EventLog";

export type JobResult = {
  job_id: string;
  ply_url: string | null;
  smpl_urls: string[];
  video_url: string;
  interactions: InteractionEvent[];
};

export type InteractionEvent = {
  time: number;
  frame: number;
  action: string;
  object: string;
  confidence: number;
  bbox?: number[];
};

export default function Home() {
  const [jobId, setJobId] = useState<string | null>(null);
  const [result, setResult] = useState<JobResult | null>(null);
  const [currentTime, setCurrentTime] = useState(0);
  const [statusMessages, setStatusMessages] = useState<string[]>([]);
  const videoRef = useRef<HTMLVideoElement>(null);

  const handleUploadComplete = useCallback((id: string) => {
    setJobId(id);
    setStatusMessages([]);

    // Connect to SSE for live updates
    const evtSource = new EventSource(`/api/stream-status/${id}`);
    evtSource.onmessage = (e) => {
      const data = JSON.parse(e.data);
      setStatusMessages((prev) => [...prev, data.message]);

      if (data.status === "completed") {
        evtSource.close();
        // Fetch results
        fetch(`/api/results/${id}`)
          .then((r) => r.json())
          .then(setResult);
      }
      if (data.status === "error") {
        evtSource.close();
      }
    };
  }, []);

  const loadTestResults = useCallback(() => {
    fetch("/api/results/test_run")
      .then((r) => r.json())
      .then((data) => {
        console.log("Loaded test results:", data);
        setJobId("test_run");
        setResult(data);
      })
      .catch((err) => console.error("Failed to load test results:", err));
  }, []);

  const handleTimeUpdate = useCallback(() => {
    if (videoRef.current) {
      setCurrentTime(videoRef.current.currentTime);
    }
  }, []);

  // Upload view
  if (!result) {
    return (
      <main className="flex min-h-screen flex-col items-center justify-center gap-4 p-8">
        <UploadPanel
          onUploadComplete={handleUploadComplete}
          statusMessages={statusMessages}
        />
        <button
          onClick={loadTestResults}
          className="rounded-lg border border-gray-700 px-4 py-2 text-sm text-gray-400 hover:border-gray-500 hover:text-gray-200"
        >
          Load Test Results
        </button>
      </main>
    );
  }

  // Results view — 3-panel layout
  return (
    <main className="flex h-screen gap-2 p-2">
      {/* Left: Video player */}
      <div className="flex w-1/3 flex-col rounded-lg border border-gray-800 bg-gray-900">
        <div className="border-b border-gray-800 px-3 py-2 text-sm font-medium text-gray-400">
          Input Video
        </div>
        <div className="flex flex-1 items-center justify-center p-2">
          <VideoPanel
            ref={videoRef}
            src={`/api${result.video_url}`}
            onTimeUpdate={handleTimeUpdate}
          />
        </div>
      </div>

      {/* Center: 3D canvas */}
      <div className="flex w-1/3 flex-col rounded-lg border border-gray-800 bg-gray-900">
        <div className="border-b border-gray-800 px-3 py-2 text-sm font-medium text-gray-400">
          3D Reconstruction
        </div>
        <div className="flex-1">
          <SceneCanvas
            plyUrl={result.ply_url ? `/api${result.ply_url}` : null}
            jobId={result.job_id}
            currentTime={currentTime}
            fps={60}
          />
        </div>
      </div>

      {/* Right: Event log */}
      <div className="flex w-1/3 flex-col rounded-lg border border-gray-800 bg-gray-900">
        <div className="border-b border-gray-800 px-3 py-2 text-sm font-medium text-gray-400">
          Interaction Events
        </div>
        <div className="flex-1 overflow-hidden">
          <EventLog events={result.interactions} currentTime={currentTime} />
        </div>
      </div>
    </main>
  );
}
