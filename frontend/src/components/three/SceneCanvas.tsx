"use client";

import { Canvas } from "@react-three/fiber";
import { OrbitControls, Grid } from "@react-three/drei";
import PointCloudViewer from "./PointCloudViewer";
import SMPLMeshPlayer from "./SMPLMeshPlayer";

type SceneCanvasProps = {
  plyUrl: string | null;
  smplUrls: string[];
  currentTime: number;
};

export default function SceneCanvas({
  plyUrl,
  smplUrls,
  currentTime,
}: SceneCanvasProps) {
  return (
    <Canvas
      camera={{ position: [0, 2, 5], fov: 60 }}
      className="h-full w-full"
    >
      <ambientLight intensity={0.5} />
      <directionalLight position={[5, 5, 5]} intensity={1} />

      <Grid
        infiniteGrid
        cellSize={0.5}
        sectionSize={2}
        fadeDistance={20}
        cellColor="#333"
        sectionColor="#555"
      />

      {plyUrl && <PointCloudViewer url={plyUrl} />}
      {smplUrls.length > 0 && (
        <SMPLMeshPlayer urls={smplUrls} currentTime={currentTime} />
      )}

      <OrbitControls
        makeDefault
        enableDamping
        dampingFactor={0.1}
        minDistance={1}
        maxDistance={20}
      />
    </Canvas>
  );
}
