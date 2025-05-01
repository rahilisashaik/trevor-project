"use client"

import type React from "react"

import { useState, useEffect } from "react"
import { Button } from "@/components/ui/button"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Input } from "@/components/ui/input"
import { Label } from "@/components/ui/label"
import { RadioGroup, RadioGroupItem } from "@/components/ui/radio-group"
import { Checkbox } from "@/components/ui/checkbox"
import { Tabs, TabsList, TabsTrigger } from "@/components/ui/tabs"
import { ScatterChart, Scatter, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer } from "recharts"
import { Search, TrendingUp, Eye, MousePointerClick } from "lucide-react"

// Define types for better type safety
type DocumentMetrics = {
  avgCpm: number
  impressions: number
  interactions: number
}

type Document = {
  id: number
  title: string
  searchTerm: string
  matchType: string
  evergreen: boolean
  metrics: DocumentMetrics
}

type SearchResults = {
  documents: Document[]
  predictedScores: {
    avgCpm: number
    impressions: number
    interactions: number
  }
  reasoning: string
}

type ChartDataPoint = {
  name: string
  documentId: number
  value: number
  searchTerm: string
  matchType: string
  evergreen: boolean
  avgCpm: number
  impressions: number
  interactions: number
}

// Generate mock data
const generateMockData = (count: number, searchTerm: string, matchType: string, evergreen: boolean): SearchResults => {
  return {
    documents: Array.from({ length: count }, (_, i) => ({
      id: i + 1,
      title: `Document ${i + 1}`,
      searchTerm: searchTerm || `Example Term ${i + 1}`,
      matchType: matchType || "exact",
      evergreen: evergreen,
      metrics: {
        avgCpm: Math.random() * 10 + 2,
        impressions: Math.floor(Math.random() * 10000) + 500,
        interactions: Math.floor(Math.random() * 1000) + 50,
      },
    })),
    predictedScores: {
      avgCpm: Math.random() * 5 + 5,
      impressions: Math.floor(Math.random() * 5000) + 5000,
      interactions: Math.floor(Math.random() * 500) + 500,
    },
    reasoning: searchTerm
      ? `Based on historical performance data, the search term "${searchTerm}" is predicted to perform well. ${matchType === "semantic" ? "Semantic matching shows strong relevance to high-performing content." : ""}${evergreen ? " The evergreen nature of this content suggests sustained engagement over time." : ""}`
      : "This prediction is based on content analysis, historical performance data, and audience engagement patterns. The semantic relevance score indicates strong topical alignment with user interests.",
  }
}

export default function Dashboard() {
  const [searchParams, setSearchParams] = useState({
    searchTerm: "",
    matchType: "exact",
    evergreen: false,
    numDocuments: 10,
  })

  // Initialize with mock data
  const [results, setResults] = useState<SearchResults | null>(null)
  const [activeMetric, setActiveMetric] = useState<keyof DocumentMetrics>("avgCpm")

  // Auto-populate with mock data on component mount
  useEffect(() => {
    setResults(generateMockData(10, "", "exact", false))
  }, [])

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault()
    setResults(
      generateMockData(
        searchParams.numDocuments,
        searchParams.searchTerm,
        searchParams.matchType,
        searchParams.evergreen,
      ),
    )
  }

  // Safely generate chart data with proper null checks
  const getChartData = (): ChartDataPoint[] => {
    if (!results || !Array.isArray(results.documents)) {
      return []
    }

    return results.documents.map((doc) => {
      // Ensure metrics object exists and has the active metric property
      const metricValue = doc.metrics && typeof doc.metrics === "object" ? (doc.metrics[activeMetric] ?? 0) : 0

      return {
        name: doc.title || `Document ${doc.id}`,
        documentId: doc.id,
        value: metricValue,
        searchTerm: doc.searchTerm,
        matchType: doc.matchType,
        evergreen: doc.evergreen,
        avgCpm: doc.metrics.avgCpm,
        impressions: doc.metrics.impressions,
        interactions: doc.metrics.interactions,
      }
    })
  }

  return (
    <div className="container mx-auto py-6 space-y-8">
      <Card>
        <CardHeader>
          <CardTitle className="text-2xl font-bold">Document Search Dashboard</CardTitle>
          <CardDescription>Search for documents and analyze their performance metrics</CardDescription>
        </CardHeader>
        <CardContent>
          <form onSubmit={handleSubmit} className="grid gap-6 md:grid-cols-2 lg:grid-cols-4">
            <div className="space-y-2">
              <Label htmlFor="searchTerm">Search Term</Label>
              <div className="relative">
                <Search className="absolute left-2.5 top-2.5 h-4 w-4 text-muted-foreground" />
                <Input
                  id="searchTerm"
                  placeholder="Enter search term..."
                  className="pl-8"
                  value={searchParams.searchTerm}
                  onChange={(e) => setSearchParams({ ...searchParams, searchTerm: e.target.value })}
                />
              </div>
            </div>

            <div className="space-y-2">
              <Label>Match Type</Label>
              <RadioGroup
                defaultValue="exact"
                onValueChange={(value) => setSearchParams({ ...searchParams, matchType: value })}
                className="flex space-x-4"
              >
                <div className="flex items-center space-x-2">
                  <RadioGroupItem value="exact" id="exact" />
                  <Label htmlFor="exact">Exact</Label>
                </div>
                <div className="flex items-center space-x-2">
                  <RadioGroupItem value="partial" id="partial" />
                  <Label htmlFor="partial">Partial</Label>
                </div>
                <div className="flex items-center space-x-2">
                  <RadioGroupItem value="semantic" id="semantic" />
                  <Label htmlFor="semantic">Semantic</Label>
                </div>
              </RadioGroup>
            </div>

            <div className="space-y-2">
              <Label>Evergreen</Label>
              <div className="flex items-center space-x-2 pt-2">
                <Checkbox
                  id="evergreen"
                  checked={searchParams.evergreen}
                  onCheckedChange={(checked) => setSearchParams({ ...searchParams, evergreen: checked === true })}
                />
                <Label htmlFor="evergreen">Yes</Label>
              </div>
            </div>

            <div className="space-y-2">
              <Label htmlFor="numDocuments">Number of Documents</Label>
              <div className="flex items-center space-x-2">
                <Input
                  id="numDocuments"
                  type="number"
                  min="1"
                  max="50"
                  value={searchParams.numDocuments}
                  onChange={(e) =>
                    setSearchParams({ ...searchParams, numDocuments: Number.parseInt(e.target.value) || 10 })
                  }
                />
                <Button type="submit">Search</Button>
              </div>
            </div>
          </form>
        </CardContent>
      </Card>

      {results && Array.isArray(results.documents) && results.documents.length > 0 && (
        <>
          <Card>
            <CardHeader>
              <CardTitle>Document Performance Metrics</CardTitle>
              <CardDescription>Interactive visualization of document metrics</CardDescription>
            </CardHeader>
            <CardContent>
              <Tabs
                defaultValue="avgCpm"
                onValueChange={(value) => setActiveMetric(value as keyof DocumentMetrics)}
                className="w-full"
              >
                <TabsList className="grid w-full grid-cols-3">
                  <TabsTrigger value="avgCpm">Avg CPM</TabsTrigger>
                  <TabsTrigger value="impressions">Impressions</TabsTrigger>
                  <TabsTrigger value="interactions">Interactions</TabsTrigger>
                </TabsList>
                <div className="mt-4 h-[350px]">
                  <ResponsiveContainer width="100%" height="100%">
                    <ScatterChart margin={{ top: 20, right: 20, bottom: 20, left: 40 }}>
                      <CartesianGrid strokeDasharray="3 3" />
                      <XAxis
                        type="number"
                        dataKey="documentId"
                        name="Document"
                        label={{ value: "Document ID", position: "insideBottom", offset: -10 }}
                      />
                      <YAxis type="number" dataKey="value" name="Value" />
                      <Tooltip content={<CustomTooltip />} />
                      <Scatter
                        name="Documents"
                        data={getChartData()}
                        fill={
                          activeMetric === "avgCpm" ? "#8884d8" : activeMetric === "impressions" ? "#82ca9d" : "#ff7300"
                        }
                      />
                    </ScatterChart>
                  </ResponsiveContainer>
                </div>
              </Tabs>
            </CardContent>
          </Card>

          <Card>
            <CardHeader>
              <CardTitle>Predicted Scores & Reasoning</CardTitle>
              <CardDescription>Performance predictions for your search query</CardDescription>
            </CardHeader>
            <CardContent>
              <div className="grid gap-6 md:grid-cols-2">
                <div>
                  <h3 className="text-lg font-medium mb-4">Predicted Scores</h3>
                  <div className="space-y-6">
                    <div className="flex items-center space-x-4">
                      <div className="w-12 h-12 rounded-full bg-purple-100 flex items-center justify-center text-purple-600">
                        <TrendingUp className="h-6 w-6" />
                      </div>
                      <div className="flex-1">
                        <div className="font-medium">Average CPM</div>
                        <div className="text-2xl font-bold">${results.predictedScores.avgCpm.toFixed(2)}</div>
                      </div>
                    </div>

                    <div className="flex items-center space-x-4">
                      <div className="w-12 h-12 rounded-full bg-green-100 flex items-center justify-center text-green-600">
                        <Eye className="h-6 w-6" />
                      </div>
                      <div className="flex-1">
                        <div className="font-medium">Impressions</div>
                        <div className="text-2xl font-bold">{results.predictedScores.impressions.toLocaleString()}</div>
                      </div>
                    </div>

                    <div className="flex items-center space-x-4">
                      <div className="w-12 h-12 rounded-full bg-orange-100 flex items-center justify-center text-orange-600">
                        <MousePointerClick className="h-6 w-6" />
                      </div>
                      <div className="flex-1">
                        <div className="font-medium">Interactions</div>
                        <div className="text-2xl font-bold">
                          {results.predictedScores.interactions.toLocaleString()}
                        </div>
                      </div>
                    </div>
                  </div>
                </div>

                <div>
                  <h3 className="text-lg font-medium mb-4">Reasoning</h3>
                  <div className="p-6 rounded-lg border bg-slate-50">
                    <p className="text-base">{results.reasoning}</p>
                  </div>
                </div>
              </div>
            </CardContent>
          </Card>
        </>
      )}
    </div>
  )
}

// Enhanced tooltip component that shows detailed document information
function CustomTooltip(props: any) {
  if (!props || !props.active || !props.payload || !Array.isArray(props.payload) || props.payload.length === 0) {
    return null
  }

  const payload = props.payload[0]
  if (!payload || !payload.payload) {
    return null
  }

  const data = payload.payload

  return (
    <div className="bg-white border rounded-md shadow-md p-4 text-sm w-64">
      <h4 className="font-bold text-base mb-2">{`Document ${data.documentId}`}</h4>

      <div className="space-y-2">
        <div>
          <span className="font-medium">Search Term:</span> {data.searchTerm || "N/A"}
        </div>
        <div>
          <span className="font-medium">Match Type:</span> {data.matchType || "N/A"}
        </div>
        <div>
          <span className="font-medium">Evergreen:</span> {data.evergreen ? "Yes" : "No"}
        </div>

        <hr className="my-2" />

        <div>
          <span className="font-medium">Avg CPM:</span> ${data.avgCpm?.toFixed(2) || "N/A"}
        </div>
        <div>
          <span className="font-medium">Impressions:</span> {data.impressions?.toLocaleString() || "N/A"}
        </div>
        <div>
          <span className="font-medium">Interactions:</span> {data.interactions?.toLocaleString() || "N/A"}
        </div>
      </div>
    </div>
  )
}

