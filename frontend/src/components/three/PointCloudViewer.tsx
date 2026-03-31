"use client";

import { useEffect, useState } from "react";
import * as THREE from "three";
import { PLYLoader } from "three/examples/jsm/loaders/PLYLoader.js";

type PointCloudViewerProps = {
  url: string;
};

export default function PointCloudViewer({ url }: PointCloudViewerProps) {
  const [geometry, setGeometry] = useState<THREE.BufferGeometry | null>(null);

  useEffect(() => {
    const loader = new PLYLoader();
    loader.load(url, (geo) => {
      geo.computeVertexNormals();
      setGeometry(geo);
    });
  }, [url]);

  if (!geometry) return null;

  return (
    <points geometry={geometry}>
      <pointsMaterial
        size={0.01}
        vertexColors={geometry.hasAttribute("color")}
        sizeAttenuation
        color={geometry.hasAttribute("color") ? undefined : "#88ccff"}
      />
    </points>
  );
}
