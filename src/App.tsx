/**
 * @license
 * SPDX-License-Identifier: Apache-2.0
 */
import React from 'react';
import LiveHUD from './components/LiveHUD';
import { Camera } from 'lucide-react';

export default function App() {
  return (
    <div className="min-h-screen bg-zinc-950 text-zinc-100 p-4 sm:p-8 font-sans">
      <div className="max-w-4xl mx-auto space-y-8">
        <header className="text-center space-y-2">
          <h1 className="text-4xl sm:text-5xl font-black tracking-tighter text-white italic uppercase">
            PGA<span className="text-emerald-500">COACH</span>.AI
          </h1>
          <p className="text-zinc-500 font-medium tracking-wide uppercase text-xs sm:text-sm">
            Biomechanical Analysis
          </p>
        </header>
        <main className="transition-all duration-500 ease-in-out">
          <div className="animate-in fade-in slide-in-from-bottom-4 duration-500">
            <LiveHUD />
          </div>
        </main>
        <footer className="text-center pt-8 border-t border-zinc-900">
          <p className="text-zinc-600 text-[10px] uppercase tracking-[-0.2em] font-bold">
            Powered by MediaPipe
          </p>
        </footer>
      </div>
    </div>
  );
}
