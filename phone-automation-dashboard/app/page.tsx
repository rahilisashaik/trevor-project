'use client';

import { useState } from 'react';
import { Button } from "@/components/ui/button";
import { ScrollArea } from "@/components/ui/scroll-area";
import PatientList from '@/components/PatientList';
import PatientDetails from '@/components/PatientDetails';
import type { Caller } from '@/lib/supabase';

export default function Home() {
  const [selectedCaller, setSelectedCaller] = useState<Caller | null>(null);

  return (
    <main className="min-h-screen bg-background">
      <div className="flex">
        {/* Sidebar */}
        <div className="w-80 border-r min-h-screen p-4">
          <div className="flex items-center gap-2 mb-8">
            <h1 className="text-2xl font-bold">Trevor Project</h1>
          </div>
          
          <div className="space-y-2">
            <h2 className="text-lg font-semibold mb-4">Patient Information</h2>
            <ScrollArea className="h-[calc(100vh-12rem)]">
              <PatientList onSelect={setSelectedCaller} />
            </ScrollArea>
          </div>
        </div>

        {/* Main Content */}
        <div className="flex-1 p-8">
          {selectedCaller ? (
            <PatientDetails caller={selectedCaller} />
          ) : (
            <div className="flex items-center justify-center h-[calc(100vh-12rem)]">
              <p className="text-muted-foreground">Select a patient to view details</p>
            </div>
          )}
        </div>
      </div>
    </main>
  );
}
