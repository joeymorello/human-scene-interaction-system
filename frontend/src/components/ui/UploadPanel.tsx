"use client";

import { useState, useCallback } from "react";

type UploadPanelProps = {
  onUploadComplete: (jobId: string) => void;
  statusMessages: string[];
};

export default function UploadPanel({
  onUploadComplete,
  statusMessages,
}: UploadPanelProps) {
  const [uploading, setUploading] = useState(false);
  const [dragOver, setDragOver] = useState(false);

  const handleUpload = useCallback(
    async (file: File) => {
      setUploading(true);
      const formData = new FormData();
      formData.append("file", file);

      try {
        const res = await fetch("/api/upload", {
          method: "POST",
          body: formData,
        });
        const data = await res.json();
        if (data.job_id) {
          onUploadComplete(data.job_id);
        }
      } catch (err) {
        console.error("Upload failed:", err);
      } finally {
        setUploading(false);
      }
    },
    [onUploadComplete]
  );

  const handleDrop = useCallback(
    (e: React.DragEvent) => {
      e.preventDefault();
      setDragOver(false);
      const file = e.dataTransfer.files[0];
      if (file) handleUpload(file);
    },
    [handleUpload]
  );

  const handleFileSelect = useCallback(
    (e: React.ChangeEvent<HTMLInputElement>) => {
      const file = e.target.files?.[0];
      if (file) handleUpload(file);
    },
    [handleUpload]
  );

  return (
    <div className="w-full max-w-xl space-y-6">
      <div className="text-center">
        <h1 className="text-3xl font-bold tracking-tight">
          Human-Scene Interaction
        </h1>
        <p className="mt-2 text-gray-400">
          Upload a video to reconstruct 3D scene, human motion, and contact
          events
        </p>
      </div>

      {/* Drop zone */}
      <div
        onDragOver={(e) => {
          e.preventDefault();
          setDragOver(true);
        }}
        onDragLeave={() => setDragOver(false)}
        onDrop={handleDrop}
        className={`flex cursor-pointer flex-col items-center justify-center rounded-xl border-2 border-dashed p-12 transition-colors ${
          dragOver
            ? "border-blue-500 bg-blue-500/10"
            : "border-gray-700 hover:border-gray-500"
        }`}
      >
        {uploading ? (
          <p className="text-gray-300">Uploading...</p>
        ) : (
          <>
            <p className="text-lg text-gray-300">
              Drag & drop a video file here
            </p>
            <p className="mt-1 text-sm text-gray-500">or</p>
            <label className="mt-3 cursor-pointer rounded-lg bg-blue-600 px-4 py-2 text-sm font-medium hover:bg-blue-700">
              Browse Files
              <input
                type="file"
                accept="video/*"
                className="hidden"
                onChange={handleFileSelect}
              />
            </label>
            <p className="mt-3 text-xs text-gray-600">
              Supports .mp4, .mov, .avi, .webm
            </p>
          </>
        )}
      </div>

      {/* Status messages */}
      {statusMessages.length > 0 && (
        <div className="rounded-lg border border-gray-800 bg-gray-900 p-4">
          <div className="space-y-1 font-mono text-sm">
            {statusMessages.map((msg, i) => (
              <div key={i} className="text-gray-300">
                <span className="text-green-400">▸</span> {msg}
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}
