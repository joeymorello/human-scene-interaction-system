"use client";

import { Component, ReactNode } from "react";

type Props = { children: ReactNode; fallback?: ReactNode };
type State = { error: string | null };

export default class ErrorBoundary extends Component<Props, State> {
  state: State = { error: null };

  static getDerivedStateFromError(err: Error) {
    return { error: err.message };
  }

  render() {
    if (this.state.error) {
      return (
        this.props.fallback ?? (
          <div className="flex h-full items-center justify-center p-4 text-red-400">
            <div className="text-center">
              <p className="font-mono text-sm">Error: {this.state.error}</p>
              <button
                onClick={() => this.setState({ error: null })}
                className="mt-2 rounded bg-gray-800 px-3 py-1 text-xs text-gray-300"
              >
                Retry
              </button>
            </div>
          </div>
        )
      );
    }
    return this.props.children;
  }
}
