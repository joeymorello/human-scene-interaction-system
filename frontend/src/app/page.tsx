"use client";

import { useState, useRef, useCallback } from "react";
import UploadPanel from "@/components/ui/UploadPanel";
import VideoPanel from "@/components/ui/VideoPanel";
import SceneCanvas from "@/components/three/SceneCanvas";
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

  const handleTimeUpdate = useCallback(() => {
    if (videoRef.current) {
      setCurrentTime(videoRef.current.currentTime);
    }
  }, []);

  // Upload view
  if (!result) {
    return (
      <main className="flex min-h-screen items-center justify-center p-8">
        <UploadPanel
          onUploadComplete={handleUploadComplete}
          statusMessages={statusMessages}
        />
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
            smplUrls={result.smpl_urls.map((u) => `/api${u}`)}
            currentTime={currentTime}
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
