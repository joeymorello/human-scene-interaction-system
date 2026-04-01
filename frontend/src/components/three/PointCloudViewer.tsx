"use client";

import { useEffect, useState } from "react";
import * as THREE from "three";
import { PLYLoader } from "three/examples/jsm/loaders/PLYLoader.js";

type PointCloudViewerProps = {
  url: string;
};

export default function PointCloudViewer({ url }: PointCloudViewerProps) {
  const [geometry, setGeometry] = useState<THREE.BufferGeometry | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    console.log("PointCloudViewer: loading", url);
    const loader = new PLYLoader();
    loader.load(
      url,
      (geo) => {
        console.log(
          "PointCloudViewer: loaded",
          geo.attributes.position?.count,
          "points"
        );
        geo.computeVertexNormals();
        setGeometry(geo);
      },
      (progress) => {
        if (progress.total) {
          console.log(
            `PointCloudViewer: ${((progress.loaded / progress.total) * 100).toFixed(0)}%`
          );
        }
      },
      (err) => {
        console.error("PointCloudViewer: load error", err);
        setError(String(err));
      }
    );
  }, [url]);

  if (error) {
    console.error("PLY load failed:", error);
    return null;
  }

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
