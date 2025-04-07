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
import { Search, TrendingUp, Eye, MousePointerClick, Loader2 } from "lucide-react"
import { Alert, AlertDescription } from "@/components/ui/alert"

// Define types for API responses
type SearchResult = {
  search_term: string
  match_type: string
  avg_cpm: number | string
  impressions: number | string
  interactions: number | string
  conversions: number | string
}

type PredictionResponse = {
  retrieved_documents: SearchResult[]
  prediction: {
    avg_cpm: number
    impressions: number
    interactions: number
    conversions: number
    reasoning: string[]
  }
  prediction_text: string
  query_time_seconds: number
}

// Define types for chart data
type ChartDataPoint = {
  name: string
  documentId: number
  value: number
  searchTerm: string
  matchType: string
  avgCpm: number
  impressions: number
  interactions: number
  conversions: number
}

export default function Dashboard() {
  const [searchParams, setSearchParams] = useState({
    searchTerm: "",
    matchType: "Broad Match",
    evergreen: false,
    numDocuments: 20,
  })

  const [results, setResults] = useState<PredictionResponse | null>(null)
  const [activeMetric, setActiveMetric] = useState<string>("avg_cpm")
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault()
    setLoading(true)
    setError(null)

    try {
      const response = await fetch("http://localhost:8000/predict", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          query: searchParams.searchTerm,
          is_evergreen: searchParams.evergreen,
          match_type: searchParams.matchType,
          k: searchParams.numDocuments,
        }),
      })

      if (!response.ok) {
        throw new Error(`API request failed with status ${response.status}`)
      }

      const data: PredictionResponse = await response.json()
      setResults(data)
    } catch (err) {
      console.error("Error fetching prediction:", err)
      setError(err instanceof Error ? err.message : "An unknown error occurred")
    } finally {
      setLoading(false)
    }
  }

  // Convert API results to chart data format
  const getChartData = (): ChartDataPoint[] => {
    if (!results || !Array.isArray(results.retrieved_documents)) {
      return []
    }

    return results.retrieved_documents.map((doc, index) => {
      // Convert string values to numbers if needed
      const avgCpm = typeof doc.avg_cpm === "string" ? parseFloat(doc.avg_cpm) : doc.avg_cpm || 0
      const impressions = typeof doc.impressions === "string" ? parseFloat(doc.impressions) : doc.impressions || 0
      const interactions = typeof doc.interactions === "string" ? parseFloat(doc.interactions) : doc.interactions || 0
      const conversions = typeof doc.conversions === "string" ? parseFloat(doc.conversions) : doc.conversions || 0

      // Get the value for the active metric
      let metricValue = 0
      if (activeMetric === "avg_cpm") metricValue = avgCpm
      else if (activeMetric === "impressions") metricValue = impressions
      else if (activeMetric === "interactions") metricValue = interactions
      else if (activeMetric === "conversions") metricValue = conversions

      return {
        name: `Doc ${index + 1}`,
        documentId: index + 1,
        value: metricValue,
        searchTerm: doc.search_term,
        matchType: doc.match_type,
        avgCpm: avgCpm,
        impressions: impressions,
        interactions: interactions,
        conversions: conversions,
      }
    })
  }

  return (
    <div className="container mx-auto py-6 space-y-8">
      <Card>
        <CardHeader>
          <CardTitle className="text-2xl font-bold">Search Terms Prediction Dashboard</CardTitle>
          <CardDescription>
            Predict performance metrics for search terms using RAG (Retrieval-Augmented Generation)
          </CardDescription>
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
                  required
                />
              </div>
            </div>

            <div className="space-y-2">
              <Label>Match Type</Label>
              <RadioGroup
                defaultValue="Broad Match"
                onValueChange={(value) => setSearchParams({ ...searchParams, matchType: value })}
                className="flex flex-col space-y-1"
              >
                <div className="flex items-center space-x-2">
                  <RadioGroupItem value="Broad Match" id="broad" />
                  <Label htmlFor="broad">Broad Match</Label>
                </div>
                <div className="flex items-center space-x-2">
                  <RadioGroupItem value="Exact Match" id="exact" />
                  <Label htmlFor="exact">Exact Match</Label>
                </div>
                <div className="flex items-center space-x-2">
                  <RadioGroupItem value="Phrase Match" id="phrase" />
                  <Label htmlFor="phrase">Phrase Match</Label>
                </div>
                <div className="flex items-center space-x-2">
                  <RadioGroupItem value="Phrase Match (close variant)" id="phrase-variant" />
                  <Label htmlFor="phrase-variant">Phrase Match (variant)</Label>
                </div>
              </RadioGroup>
            </div>

            <div className="space-y-2">
              <Label>Content Type</Label>
              <div className="flex items-center space-x-2 pt-2">
                <Checkbox
                  id="evergreen"
                  checked={searchParams.evergreen}
                  onCheckedChange={(checked) => setSearchParams({ ...searchParams, evergreen: checked === true })}
                />
                <Label htmlFor="evergreen">Evergreen Content</Label>
              </div>
              <p className="text-sm text-muted-foreground mt-2">
                Evergreen content remains relevant for a long time, while non-evergreen content is time-sensitive.
              </p>
            </div>

            <div className="space-y-2">
              <Label htmlFor="numDocuments">Number of Similar Documents</Label>
              <div className="flex items-center space-x-2">
                <Input
                  id="numDocuments"
                  type="number"
                  min="5"
                  max="50"
                  value={searchParams.numDocuments}
                  onChange={(e) =>
                    setSearchParams({ ...searchParams, numDocuments: Number.parseInt(e.target.value) || 20 })
                  }
                />
                <Button type="submit" disabled={loading}>
                  {loading ? (
                    <>
                      <Loader2 className="mr-2 h-4 w-4 animate-spin" />
                      Predicting
                    </>
                  ) : (
                    "Predict"
                  )}
                </Button>
              </div>
            </div>
          </form>

          {error && (
            <Alert variant="destructive" className="mt-4">
              <AlertDescription>{error}</AlertDescription>
            </Alert>
          )}
        </CardContent>
      </Card>

      {results && Array.isArray(results.retrieved_documents) && results.retrieved_documents.length > 0 && (
        <>
          <Card>
            <CardHeader>
              <CardTitle>Similar Search Terms Performance</CardTitle>
              <CardDescription>
                Visualization of similar search terms and their performance metrics (used for prediction)
              </CardDescription>
            </CardHeader>
            <CardContent>
              <Tabs defaultValue="avg_cpm" onValueChange={setActiveMetric} className="w-full">
                <TabsList className="grid w-full grid-cols-4">
                  <TabsTrigger value="avg_cpm">Avg CPM</TabsTrigger>
                  <TabsTrigger value="impressions">Impressions</TabsTrigger>
                  <TabsTrigger value="interactions">Interactions</TabsTrigger>
                  <TabsTrigger value="conversions">Conversions</TabsTrigger>
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
                          activeMetric === "avg_cpm"
                            ? "#8884d8"
                            : activeMetric === "impressions"
                            ? "#82ca9d"
                            : activeMetric === "interactions"
                            ? "#ff7300"
                            : "#ff5252"
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
              <CardTitle>Predicted Metrics & Reasoning</CardTitle>
              <CardDescription>
                Performance predictions for "{searchParams.searchTerm}" ({searchParams.matchType})
              </CardDescription>
            </CardHeader>
            <CardContent>
              <div className="grid gap-6 md:grid-cols-2">
                <div>
                  <h3 className="text-lg font-medium mb-4">Predicted Metrics</h3>
                  <div className="space-y-6">
                    <div className="flex items-center space-x-4">
                      <div className="w-12 h-12 rounded-full bg-purple-100 flex items-center justify-center text-purple-600">
                        <TrendingUp className="h-6 w-6" />
                      </div>
                      <div className="flex-1">
                        <div className="font-medium">Average CPM</div>
                        <div className="text-2xl font-bold">
                          ${results.prediction.avg_cpm ? results.prediction.avg_cpm.toFixed(2) : "N/A"}
                        </div>
                      </div>
                    </div>

                    <div className="flex items-center space-x-4">
                      <div className="w-12 h-12 rounded-full bg-green-100 flex items-center justify-center text-green-600">
                        <Eye className="h-6 w-6" />
                      </div>
                      <div className="flex-1">
                        <div className="font-medium">Impressions</div>
                        <div className="text-2xl font-bold">
                          {results.prediction.impressions
                            ? Math.round(results.prediction.impressions).toLocaleString()
                            : "N/A"}
                        </div>
                      </div>
                    </div>

                    <div className="flex items-center space-x-4">
                      <div className="w-12 h-12 rounded-full bg-orange-100 flex items-center justify-center text-orange-600">
                        <MousePointerClick className="h-6 w-6" />
                      </div>
                      <div className="flex-1">
                        <div className="font-medium">Interactions</div>
                        <div className="text-2xl font-bold">
                          {results.prediction.interactions
                            ? Math.round(results.prediction.interactions).toLocaleString()
                            : "N/A"}
                        </div>
                      </div>
                    </div>
                  </div>
                </div>

                <div>
                  <h3 className="text-lg font-medium mb-4">Reasoning</h3>
                  <div className="p-6 rounded-lg border bg-slate-50">
                    {results.prediction.reasoning && results.prediction.reasoning.length > 0 ? (
                      <ul className="list-disc pl-5 space-y-2">
                        {results.prediction.reasoning.map((reason, index) => (
                          <li key={index}>{reason.replace("• ", "")}</li>
                        ))}
                      </ul>
                    ) : (
                      <p className="text-base">{results.prediction_text}</p>
                    )}
                  </div>
                  <div className="mt-4 text-sm text-muted-foreground">
                    <p>Query time: {results.query_time_seconds.toFixed(2)} seconds</p>
                    <p>Retrieved {results.retrieved_documents.length} similar search terms for analysis</p>
                  </div>
                </div>
              </div>
            </CardContent>
          </Card>

          <Card>
            <CardHeader>
              <CardTitle>Similar Search Terms</CardTitle>
              <CardDescription>Top similar search terms used for prediction</CardDescription>
            </CardHeader>
            <CardContent>
              <div className="overflow-x-auto">
                <table className="w-full border-collapse">
                  <thead>
                    <tr className="border-b">
                      <th className="text-left py-2 px-4">#</th>
                      <th className="text-left py-2 px-4">Search Term</th>
                      <th className="text-left py-2 px-4">Match Type</th>
                      <th className="text-right py-2 px-4">Avg CPM</th>
                      <th className="text-right py-2 px-4">Impressions</th>
                      <th className="text-right py-2 px-4">Interactions</th>
                    </tr>
                  </thead>
                  <tbody>
                    {results.retrieved_documents.slice(0, 10).map((doc, index) => (
                      <tr key={index} className="border-b hover:bg-slate-50">
                        <td className="py-2 px-4">{index + 1}</td>
                        <td className="py-2 px-4">{doc.search_term}</td>
                        <td className="py-2 px-4">{doc.match_type}</td>
                        <td className="py-2 px-4 text-right">
                          {typeof doc.avg_cpm === "number"
                            ? `$${doc.avg_cpm.toFixed(2)}`
                            : typeof doc.avg_cpm === "string"
                            ? `$${doc.avg_cpm}`
                            : "N/A"}
                        </td>
                        <td className="py-2 px-4 text-right">
                          {typeof doc.impressions === "number"
                            ? doc.impressions.toLocaleString()
                            : typeof doc.impressions === "string"
                            ? parseInt(doc.impressions).toLocaleString()
                            : "N/A"}
                        </td>
                        <td className="py-2 px-4 text-right">
                          {typeof doc.interactions === "number"
                            ? doc.interactions.toLocaleString()
                            : typeof doc.interactions === "string"
                            ? parseInt(doc.interactions).toLocaleString()
                            : "N/A"}
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
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
        <div>
          <span className="font-medium">Conversions:</span> {data.conversions?.toLocaleString() || "N/A"}
        </div>
      </div>
    </div>
  )
}

