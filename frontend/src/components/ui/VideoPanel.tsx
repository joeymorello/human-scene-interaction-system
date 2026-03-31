"use client";

import { forwardRef } from "react";

type VideoPanelProps = {
  src: string;
  onTimeUpdate: () => void;
};

const VideoPanel = forwardRef<HTMLVideoElement, VideoPanelProps>(
  ({ src, onTimeUpdate }, ref) => {
    return (
      <video
        ref={ref}
        src={src}
        controls
        onTimeUpdate={onTimeUpdate}
        className="max-h-full w-full rounded object-contain"
      />
    );
  }
);

VideoPanel.displayName = "VideoPanel";
export default VideoPanel;
