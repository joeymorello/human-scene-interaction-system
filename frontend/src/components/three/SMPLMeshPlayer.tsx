"use client";

import { useEffect, useMemo, useRef, useState } from "react";
import * as THREE from "three";

type SMPLMeshPlayerProps = {
  jobId: string;
  currentTime: number;
  fps?: number;
};

type SMPLData = {
  faces: Uint32Array;
  frames: Float32Array[]; // each is V*3 floats
  numVerts: number;
};

export default function SMPLMeshPlayer({
  jobId,
  currentTime,
  fps = 30,
}: SMPLMeshPlayerProps) {
  const meshRef = useRef<THREE.Mesh>(null);
  const [data, setData] = useState<SMPLData | null>(null);

  // Load binary SMPL data
  useEffect(() => {
    if (!jobId) return;

    fetch(`/api/results/${jobId}/smpl.bin`)
      .then((r) => r.arrayBuffer())
      .then((buf) => {
        const view = new DataView(buf);
        const numFrames = view.getUint32(0, true);
        const numVerts = view.getUint32(4, true);
        const numFaces = view.getUint32(8, true);

        const headerSize = 12;
        const facesSize = numFaces * 3 * 4;
        const faces = new Uint32Array(buf, headerSize, numFaces * 3);

        const vertSize = numVerts * 3 * 4;
        const framesStart = headerSize + facesSize;
        const frames: Float32Array[] = [];
        for (let i = 0; i < numFrames; i++) {
          const offset = framesStart + i * vertSize;
          frames.push(new Float32Array(buf, offset, numVerts * 3));
        }

        console.log(
          `SMPLMeshPlayer: loaded ${numFrames} frames, ${numVerts} verts, ${numFaces} faces`
        );
        setData({ faces, frames, numVerts });
      })
      .catch((err) => console.error("Failed to load SMPL data:", err));
  }, [jobId]);

  // Select frame based on currentTime
  const frameIndex = useMemo(() => {
    if (!data || data.frames.length === 0) return -1;
    const idx = Math.floor(currentTime * fps);
    return Math.max(0, Math.min(idx, data.frames.length - 1));
  }, [currentTime, fps, data]);

  // Update mesh geometry when frame changes
  useEffect(() => {
    if (frameIndex < 0 || !meshRef.current || !data) return;

    const geo = meshRef.current.geometry;
    geo.setAttribute(
      "position",
      new THREE.BufferAttribute(data.frames[frameIndex], 3)
    );
    if (!geo.index) {
      geo.setIndex(new THREE.BufferAttribute(data.faces, 1));
    }
    geo.computeVertexNormals();
    geo.attributes.position.needsUpdate = true;
  }, [frameIndex, data]);

  if (!data) return null;

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
