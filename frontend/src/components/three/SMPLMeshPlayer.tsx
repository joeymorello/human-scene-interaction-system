"use client";

import { useEffect, useMemo, useRef, useState } from "react";
import * as THREE from "three";

type SMPLMeshPlayerProps = {
  urls: string[];
  currentTime: number;
  fps?: number;
};

type SMPLFrame = {
  vertices: Float32Array;
  faces: Uint32Array;
};

export default function SMPLMeshPlayer({
  urls,
  currentTime,
  fps = 30,
}: SMPLMeshPlayerProps) {
  const meshRef = useRef<THREE.Mesh>(null);
  const [frames, setFrames] = useState<SMPLFrame[]>([]);

  // Load all SMPL .npz frames
  useEffect(() => {
    if (urls.length === 0) return;

    // TODO: Load .npz files — requires a browser-side NPZ parser
    // For now, frames are loaded as binary ArrayBuffers and parsed
    // Format expected: { vertices: Float32Array(V*3), faces: Uint32Array(F*3) }
    //
    // In production, the backend should convert .npz to a browser-friendly
    // format (e.g., GLB or a flat binary with a header).
    console.log(`SMPLMeshPlayer: ${urls.length} frames to load`);
  }, [urls]);

  // Select frame based on currentTime
  const frameIndex = useMemo(() => {
    if (frames.length === 0) return -1;
    const idx = Math.floor(currentTime * fps);
    return Math.min(idx, frames.length - 1);
  }, [currentTime, fps, frames.length]);

  // Update mesh geometry when frame changes
  useEffect(() => {
    if (frameIndex < 0 || !meshRef.current || frames.length === 0) return;

    const frame = frames[frameIndex];
    const geo = meshRef.current.geometry;

    geo.setAttribute(
      "position",
      new THREE.BufferAttribute(frame.vertices, 3)
    );
    geo.setIndex(new THREE.BufferAttribute(frame.faces, 1));
    geo.computeVertexNormals();
    geo.attributes.position.needsUpdate = true;
  }, [frameIndex, frames]);

  return (
    <mesh ref={meshRef}>
      <bufferGeometry />
      <meshStandardMaterial
        color="#ff6b6b"
        transparent
        opacity={0.85}
        side={THREE.DoubleSide}
      />
    </mesh>
  );
}
