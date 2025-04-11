'use client';

import { useState } from 'react';
import { Button } from "@/components/ui/button";
import { ScrollArea } from "@/components/ui/scroll-area";
import PatientList from '@/components/PatientList';
import PatientDetails from '@/components/PatientDetails';
import type { Caller } from '@/lib/supabase';
import { RefreshCw } from "lucide-react";

export default function Home() {
  const [selectedCaller, setSelectedCaller] = useState<Caller | null>(null);
  const [refreshKey, setRefreshKey] = useState(0);

  const handleRefresh = () => {
    setRefreshKey(prev => prev + 1);
  };

  return (
    <main className="min-h-screen bg-background">
      <div className="flex">
        {/* Sidebar */}
        <div className="w-80 border-r min-h-screen p-4">
          <div className="flex items-center gap-2 mb-8">
            <h1 className="text-2xl font-bold">Trevor Project</h1>
          </div>
          
          <div className="space-y-2">
            <div className="flex items-center justify-between">
              <h2 className="text-lg font-semibold">Patient Information</h2>
              <Button 
                variant="outline" 
                size="sm"
                onClick={handleRefresh}
                className="flex items-center gap-2"
              >
                <RefreshCw className="h-4 w-4" />
                Refresh
              </Button>
            </div>
            <ScrollArea className="h-[calc(100vh-12rem)]">
              <PatientList key={refreshKey} onSelect={setSelectedCaller} />
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
