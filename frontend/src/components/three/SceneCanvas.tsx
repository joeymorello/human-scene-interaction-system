"use client";

import { Suspense } from "react";
import { Canvas } from "@react-three/fiber";
import { OrbitControls, Grid } from "@react-three/drei";
import ErrorBoundary from "@/components/ErrorBoundary";
import PointCloudViewer from "./PointCloudViewer";
import SMPLMeshPlayer from "./SMPLMeshPlayer";

type SceneCanvasProps = {
  plyUrl: string | null;
  jobId: string;
  currentTime: number;
  fps?: number;
};

export default function SceneCanvas({
  plyUrl,
  jobId,
  currentTime,
  fps,
}: SceneCanvasProps) {
  return (
    <ErrorBoundary>
      <Canvas
        camera={{ position: [0, -3, -2], fov: 60, up: [0, -1, 0], near: 0.01 }}
        className="h-full w-full"
      >
        <ambientLight intensity={0.6} />
        <directionalLight position={[2, -5, -3]} intensity={1} />

        <Suspense fallback={null}>
          {plyUrl && <PointCloudViewer url={plyUrl} />}
          <SMPLMeshPlayer jobId={jobId} currentTime={currentTime} fps={fps} />
        </Suspense>

        <OrbitControls
          makeDefault
          enableDamping
          dampingFactor={0.1}
          target={[0, 0, 1.2]}
          minDistance={0.5}
          maxDistance={20}
        />
      </Canvas>
    </ErrorBoundary>
  );
}
