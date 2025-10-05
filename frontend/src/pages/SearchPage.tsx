import React, { useState } from 'react'
import { useMutation } from 'react-query'
import axios from 'axios'
import toast from 'react-hot-toast'
import {
  MagnifyingGlassIcon,
  DocumentTextIcon,
  SparklesIcon,
  ClockIcon,
  StarIcon,
} from '@heroicons/react/24/outline'

interface SearchResult {
  resume_id: number
  filename: string
  similarity_score: number
  evidence_snippets: string[]
  created_at: string
}

const SearchPage: React.FC = () => {
  const [query, setQuery] = useState('')
  const [results, setResults] = useState<SearchResult[]>([])
  const [totalSearched, setTotalSearched] = useState(0)
  const [searchHistory, setSearchHistory] = useState<string[]>([])

  const searchMutation = useMutation(
    async (searchData: { query: string; k: number }) => {
      const response = await axios.post('/api/ask', searchData)
      return response.data
    },
    {
      onSuccess: (data) => {
        setResults(data.answers)
        setTotalSearched(data.total_resumes_searched)
        
        // Add to search history
        if (query.trim() && !searchHistory.includes(query.trim())) {
          setSearchHistory(prev => [query.trim(), ...prev.slice(0, 4)])
        }
        
        toast.success(`Found ${data.answers.length} relevant results`)
      },
      onError: (error: any) => {
        console.error('Search error:', error)
        const errorMessage = error.response?.data?.error?.message || 'Search failed'
        toast.error(errorMessage)
      },
    }
  )

  const handleSearch = (e: React.FormEvent) => {
    e.preventDefault()
    if (!query.trim()) {
      toast.error('Please enter a search query')
      return
    }
    searchMutation.mutate({ query: query.trim(), k: 10 })
  }

  const handleQuickSearch = (quickQuery: string) => {
    setQuery(quickQuery)
    searchMutation.mutate({ query: quickQuery, k: 10 })
  }

  const quickSearches = [
    'Python developer with machine learning experience',
    'Frontend developer with React and TypeScript',
    'Data scientist with PhD',
    'DevOps engineer with AWS and Kubernetes',
    'Product manager with 5+ years experience',
    'Full stack developer with Node.js',
  ]

  const getSimilarityColor = (score: number) => {
    if (score >= 0.8) return 'text-green-600 bg-green-100'
    if (score >= 0.6) return 'text-yellow-600 bg-yellow-100'
    return 'text-red-600 bg-red-100'
  }

  const formatScore = (score: number) => {
    return Math.round(score * 100)
  }

  return (
    <div className="space-y-8">
      <div>
        <h1 className="text-3xl font-bold text-gray-900">Search Resumes</h1>
        <p className="mt-2 text-gray-600">
          Ask natural language questions to find relevant resume content using AI-powered semantic search.
        </p>
      </div>

      {/* Search Form */}
      <div className="card">
        <div className="p-6">
          <form onSubmit={handleSearch} className="space-y-4">
            <div className="relative">
              <MagnifyingGlassIcon className="absolute left-3 top-1/2 transform -translate-y-1/2 h-5 w-5 text-gray-400" />
              <input
                type="text"
                value={query}
                onChange={(e) => setQuery(e.target.value)}
                className="input-field pl-10"
                placeholder="e.g., Find developers with Python and machine learning experience..."
              />
            </div>
            <div className="flex items-center justify-between">
              <p className="text-sm text-gray-500">
                Search across {totalSearched || 'your'} resume{totalSearched !== 1 ? 's' : ''}
              </p>
              <button
                type="submit"
                disabled={searchMutation.isLoading}
                className="btn-primary disabled:opacity-50 disabled:cursor-not-allowed"
              >
                {searchMutation.isLoading ? (
                  <div className="flex items-center">
                    <div className="animate-spin rounded-full h-5 w-5 border-b-2 border-white mr-2"></div>
                    Searching...
                  </div>
                ) : (
                  <div className="flex items-center">
                    <SparklesIcon className="h-5 w-5 mr-2" />
                    Search with AI
                  </div>
                )}
              </button>
            </div>
          </form>
        </div>
      </div>

      {/* Quick Searches */}
      {results.length === 0 && (
        <div className="card">
          <div className="p-6">
            <h3 className="text-lg font-medium text-gray-900 mb-4">Try these example searches:</h3>
            <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
              {quickSearches.map((quickQuery, index) => (
                <button
                  key={index}
                  onClick={() => handleQuickSearch(quickQuery)}
                  className="text-left p-3 rounded-lg border border-gray-200 hover:border-primary-300 hover:bg-primary-50 transition-all duration-200"
                >
                  <span className="text-sm text-gray-700">{quickQuery}</span>
                </button>
              ))}
            </div>
          </div>
        </div>
      )}

      {/* Search History */}
      {searchHistory.length > 0 && results.length === 0 && (
        <div className="card">
          <div className="p-6">
            <h3 className="text-lg font-medium text-gray-900 mb-4 flex items-center">
              <ClockIcon className="h-5 w-5 mr-2 text-gray-400" />
              Recent Searches
            </h3>
            <div className="space-y-2">
              {searchHistory.map((historyQuery, index) => (
                <button
                  key={index}
                  onClick={() => handleQuickSearch(historyQuery)}
                  className="text-left p-2 rounded-lg hover:bg-gray-50 transition-colors duration-200 text-sm text-gray-600 w-full"
                >
                  {historyQuery}
                </button>
              ))}
            </div>
          </div>
        </div>
      )}

      {/* Search Results */}
      {results.length > 0 && (
        <div className="space-y-6">
          <div className="flex items-center justify-between">
            <h2 className="text-2xl font-bold text-gray-900">
              Search Results ({results.length})
            </h2>
            <p className="text-sm text-gray-500">
              Searched {totalSearched} resume{totalSearched !== 1 ? 's' : ''}
            </p>
          </div>

          <div className="space-y-4">
            {results.map((result, index) => (
              <div key={result.resume_id} className="card hover:shadow-md transition-shadow duration-200">
                <div className="p-6">
                  <div className="flex items-start justify-between">
                    <div className="flex items-start space-x-4 flex-1">
                      <DocumentTextIcon className="h-8 w-8 text-gray-400 mt-1" />
                      <div className="flex-1">
                        <div className="flex items-center space-x-3">
                          <h3 className="text-lg font-medium text-gray-900">
                            {result.filename}
                          </h3>
                          <span className={`px-2 py-1 text-xs font-medium rounded-full ${getSimilarityColor(result.similarity_score)}`}>
                            {formatScore(result.similarity_score)}% match
                          </span>
                        </div>
                        <p className="text-sm text-gray-500 mt-1">
                          Uploaded {new Date(result.created_at).toLocaleDateString()}
                        </p>
                      </div>
                    </div>
                    <button className="text-primary-600 hover:text-primary-700 transition-colors duration-200">
                      <StarIcon className="h-5 w-5" />
                    </button>
                  </div>

                  {result.evidence_snippets.length > 0 && (
                    <div className="mt-4">
                      <h4 className="text-sm font-medium text-gray-900 mb-2">Relevant excerpts:</h4>
                      <div className="space-y-2">
                        {result.evidence_snippets.map((snippet, snippetIndex) => (
                          <div
                            key={snippetIndex}
                            className="bg-yellow-50 border border-yellow-200 rounded-lg p-3"
                          >
                            <p className="text-sm text-gray-700">
                              ...{snippet.trim()}...
                            </p>
                          </div>
                        ))}
                      </div>
                    </div>
                  )}

                  <div className="mt-4 flex items-center justify-between">
                    <div className="flex items-center space-x-4 text-sm text-gray-500">
                      <span>Resume ID: {result.resume_id}</span>
                      <span>•</span>
                      <span>Similarity: {(result.similarity_score * 100).toFixed(1)}%</span>
                    </div>
                    <button
                      onClick={() => window.open(`/candidates/${result.resume_id}`, '_blank')}
                      className="text-sm text-primary-600 hover:text-primary-700 font-medium"
                    >
                      View Full Resume →
                    </button>
                  </div>
                </div>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* No Results */}
      {results.length === 0 && searchMutation.isSuccess && (
        <div className="card">
          <div className="p-8 text-center">
            <MagnifyingGlassIcon className="mx-auto h-12 w-12 text-gray-400" />
            <h3 className="mt-4 text-lg font-medium text-gray-900">No results found</h3>
            <p className="mt-2 text-gray-500">
              Try adjusting your search query or check if you have uploaded resumes.
            </p>
          </div>
        </div>
      )}
    </div>
  )
}

export default SearchPage
