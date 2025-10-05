import React, { useState } from 'react'
import { useQuery, useMutation, useQueryClient } from 'react-query'
import axios from 'axios'
import toast from 'react-hot-toast'
import {
  BriefcaseIcon,
  PlusIcon,
  BuildingOfficeIcon,
  UserGroupIcon,
  SparklesIcon,
  XMarkIcon,
} from '@heroicons/react/24/outline'

interface Job {
  id: number
  title: string
  description: string
  requirements: string[]
  company: string
  created_at: string
}

interface JobMatch {
  resume_id: number
  candidate_name: string
  match_percentage: number
  similarity_score: number
  matching_skills: string[]
  missing_requirements: string[]
  evidence_snippets: string[]
}

const JobsPage: React.FC = () => {
  const [showCreateForm, setShowCreateForm] = useState(false)
  const [selectedJob, setSelectedJob] = useState<Job | null>(null)
  const [matches, setMatches] = useState<JobMatch[]>([])
  const [formData, setFormData] = useState({
    title: '',
    description: '',
    company: '',
    requirements: ['']
  })

  const queryClient = useQueryClient()

  const { data: jobs, isLoading: jobsLoading } = useQuery('jobs', async () => {
    // Since we don't have a GET /api/jobs endpoint, we'll simulate with empty array
    // In a real implementation, you'd fetch from GET /api/jobs
    return []
  })

  const createJobMutation = useMutation(
    async (jobData: any) => {
      const response = await axios.post('/api/jobs', jobData, {
        headers: {
          'Idempotency-Key': `job_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`
        }
      })
      return response.data
    },
    {
      onSuccess: () => {
        toast.success('Job created successfully!')
        setShowCreateForm(false)
        setFormData({ title: '', description: '', company: '', requirements: [''] })
        queryClient.invalidateQueries('jobs')
      },
      onError: (error: any) => {
        const errorMessage = error.response?.data?.error?.message || 'Failed to create job'
        toast.error(errorMessage)
      }
    }
  )

  const matchJobMutation = useMutation(
    async ({ jobId, topN }: { jobId: number, topN: number }) => {
      const response = await axios.post(`/api/jobs/${jobId}/match`, { top_n: topN })
      return response.data
    },
    {
      onSuccess: (data) => {
        setMatches(data.matches)
        toast.success(`Found ${data.matches.length} matching candidates`)
      },
      onError: (error: any) => {
        const errorMessage = error.response?.data?.error?.message || 'Failed to match candidates'
        toast.error(errorMessage)
      }
    }
  )

  const handleCreateJob = (e: React.FormEvent) => {
    e.preventDefault()
    
    if (!formData.title.trim() || !formData.company.trim() || !formData.description.trim()) {
      toast.error('Please fill in all required fields')
      return
    }

    const requirements = formData.requirements.filter(req => req.trim() !== '')
    if (requirements.length === 0) {
      toast.error('Please add at least one requirement')
      return
    }

    createJobMutation.mutate({
      title: formData.title.trim(),
      description: formData.description.trim(),
      company: formData.company.trim(),
      requirements
    })
  }

  const addRequirement = () => {
    setFormData(prev => ({
      ...prev,
      requirements: [...prev.requirements, '']
    }))
  }

  const updateRequirement = (index: number, value: string) => {
    setFormData(prev => ({
      ...prev,
      requirements: prev.requirements.map((req, i) => i === index ? value : req)
    }))
  }

  const removeRequirement = (index: number) => {
    setFormData(prev => ({
      ...prev,
      requirements: prev.requirements.filter((_, i) => i !== index)
    }))
  }

  const handleMatchCandidates = (job: Job) => {
    setSelectedJob(job)
    matchJobMutation.mutate({ jobId: job.id, topN: 10 })
  }

  const getMatchColor = (percentage: number) => {
    if (percentage >= 80) return 'text-green-600 bg-green-100'
    if (percentage >= 60) return 'text-yellow-600 bg-yellow-100'
    return 'text-red-600 bg-red-100'
  }

  return (
    <div className="space-y-8">
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-3xl font-bold text-gray-900">Job Management</h1>
          <p className="mt-2 text-gray-600">
            Create job postings and find the best matching candidates using AI.
          </p>
        </div>
        <button
          onClick={() => setShowCreateForm(true)}
          className="btn-primary"
        >
          <PlusIcon className="h-5 w-5 mr-2" />
          Create Job
        </button>
      </div>

      {/* Create Job Form */}
      {showCreateForm && (
        <div className="fixed inset-0 z-50 overflow-y-auto">
          <div className="flex items-center justify-center min-h-screen px-4">
            <div className="fixed inset-0 bg-gray-500 bg-opacity-75" onClick={() => setShowCreateForm(false)}></div>
            <div className="relative bg-white rounded-lg p-6 w-full max-w-2xl">
              <div className="flex items-center justify-between mb-6">
                <h3 className="text-lg font-medium text-gray-900">Create New Job</h3>
                <button
                  onClick={() => setShowCreateForm(false)}
                  className="text-gray-400 hover:text-gray-500"
                >
                  <XMarkIcon className="h-6 w-6" />
                </button>
              </div>

              <form onSubmit={handleCreateJob} className="space-y-6">
                <div className="grid grid-cols-2 gap-4">
                  <div>
                    <label className="block text-sm font-medium text-gray-700">Job Title *</label>
                    <input
                      type="text"
                      value={formData.title}
                      onChange={(e) => setFormData(prev => ({ ...prev, title: e.target.value }))}
                      className="mt-1 input-field"
                      placeholder="e.g., Senior Software Engineer"
                    />
                  </div>
                  <div>
                    <label className="block text-sm font-medium text-gray-700">Company *</label>
                    <input
                      type="text"
                      value={formData.company}
                      onChange={(e) => setFormData(prev => ({ ...prev, company: e.target.value }))}
                      className="mt-1 input-field"
                      placeholder="e.g., Tech Corp"
                    />
                  </div>
                </div>

                <div>
                  <label className="block text-sm font-medium text-gray-700">Job Description *</label>
                  <textarea
                    value={formData.description}
                    onChange={(e) => setFormData(prev => ({ ...prev, description: e.target.value }))}
                    rows={4}
                    className="mt-1 input-field"
                    placeholder="Describe the role, responsibilities, and what you're looking for..."
                  />
                </div>

                <div>
                  <label className="block text-sm font-medium text-gray-700 mb-2">Requirements *</label>
                  {formData.requirements.map((requirement, index) => (
                    <div key={index} className="flex items-center space-x-2 mb-2">
                      <input
                        type="text"
                        value={requirement}
                        onChange={(e) => updateRequirement(index, e.target.value)}
                        className="flex-1 input-field"
                        placeholder="e.g., 3+ years Python experience"
                      />
                      {formData.requirements.length > 1 && (
                        <button
                          type="button"
                          onClick={() => removeRequirement(index)}
                          className="text-red-500 hover:text-red-700"
                        >
                          <XMarkIcon className="h-5 w-5" />
                        </button>
                      )}
                    </div>
                  ))}
                  <button
                    type="button"
                    onClick={addRequirement}
                    className="text-primary-600 hover:text-primary-700 text-sm font-medium"
                  >
                    + Add Requirement
                  </button>
                </div>

                <div className="flex justify-end space-x-3">
                  <button
                    type="button"
                    onClick={() => setShowCreateForm(false)}
                    className="btn-secondary"
                  >
                    Cancel
                  </button>
                  <button
                    type="submit"
                    disabled={createJobMutation.isLoading}
                    className="btn-primary disabled:opacity-50"
                  >
                    {createJobMutation.isLoading ? 'Creating...' : 'Create Job'}
                  </button>
                </div>
              </form>
            </div>
          </div>
        </div>
      )}

      {/* Jobs List */}
      <div className="grid grid-cols-1 gap-6">
        {jobs?.length === 0 && !jobsLoading && (
          <div className="card">
            <div className="p-8 text-center">
              <BriefcaseIcon className="mx-auto h-12 w-12 text-gray-400" />
              <h3 className="mt-4 text-lg font-medium text-gray-900">No jobs yet</h3>
              <p className="mt-2 text-gray-500">
                Create your first job posting to start finding matching candidates.
              </p>
              <button
                onClick={() => setShowCreateForm(true)}
                className="mt-4 btn-primary"
              >
                Create Your First Job
              </button>
            </div>
          </div>
        )}

        {/* Demo Job for Testing */}
        <div className="card">
          <div className="p-6">
            <div className="flex items-start justify-between">
              <div className="flex items-start space-x-4">
                <BriefcaseIcon className="h-8 w-8 text-primary-600 mt-1" />
                <div>
                  <h3 className="text-xl font-semibold text-gray-900">Senior Python Developer</h3>
                  <div className="flex items-center space-x-2 mt-1">
                    <BuildingOfficeIcon className="h-4 w-4 text-gray-400" />
                    <span className="text-gray-600">TechCorp Inc.</span>
                  </div>
                  <p className="text-gray-600 mt-2">
                    We're looking for an experienced Python developer to join our machine learning team. 
                    You'll work on cutting-edge AI projects and help build scalable data pipelines.
                  </p>
                  <div className="mt-4">
                    <h4 className="text-sm font-medium text-gray-900 mb-2">Requirements:</h4>
                    <div className="flex flex-wrap gap-2">
                      {['Python', 'Machine Learning', 'TensorFlow', 'AWS', 'Docker'].map((req) => (
                        <span key={req} className="px-2 py-1 bg-gray-100 text-gray-700 rounded text-sm">
                          {req}
                        </span>
                      ))}
                    </div>
                  </div>
                </div>
              </div>
            </div>
            <div className="mt-6 flex items-center justify-between">
              <span className="text-sm text-gray-500">Demo Job • Created for testing</span>
              <button
                onClick={() => handleMatchCandidates({
                  id: 1,
                  title: 'Senior Python Developer',
                  description: 'Python developer for ML team',
                  requirements: ['Python', 'Machine Learning', 'TensorFlow', 'AWS', 'Docker'],
                  company: 'TechCorp Inc.',
                  created_at: new Date().toISOString()
                })}
                disabled={matchJobMutation.isLoading}
                className="btn-primary disabled:opacity-50"
              >
                {matchJobMutation.isLoading ? (
                  <div className="flex items-center">
                    <div className="animate-spin rounded-full h-4 w-4 border-b-2 border-white mr-2"></div>
                    Finding Candidates...
                  </div>
                ) : (
                  <div className="flex items-center">
                    <SparklesIcon className="h-4 w-4 mr-2" />
                    Find Candidates
                  </div>
                )}
              </button>
            </div>
          </div>
        </div>
      </div>

      {/* Matching Results */}
      {selectedJob && matches.length > 0 && (
        <div className="space-y-6">
          <div className="flex items-center justify-between">
            <h2 className="text-2xl font-bold text-gray-900">
              Matching Candidates for "{selectedJob.title}"
            </h2>
            <span className="text-sm text-gray-500">
              {matches.length} candidates found
            </span>
          </div>

          <div className="space-y-4">
            {matches.map((match) => (
              <div key={match.resume_id} className="card hover:shadow-md transition-shadow">
                <div className="p-6">
                  <div className="flex items-start justify-between">
                    <div className="flex-1">
                      <div className="flex items-center space-x-3">
                        <h3 className="text-lg font-medium text-gray-900">
                          {match.candidate_name}
                        </h3>
                        <span className={`px-3 py-1 text-sm font-medium rounded-full ${getMatchColor(match.match_percentage)}`}>
                          {match.match_percentage}% Match
                        </span>
                      </div>

                      <div className="mt-3 grid grid-cols-2 gap-4">
                        <div>
                          <h4 className="text-sm font-medium text-gray-900 mb-2">Matching Skills:</h4>
                          <div className="flex flex-wrap gap-1">
                            {match.matching_skills.map((skill) => (
                              <span key={skill} className="px-2 py-1 bg-green-100 text-green-800 rounded text-xs">
                                {skill}
                              </span>
                            ))}
                          </div>
                        </div>

                        <div>
                          <h4 className="text-sm font-medium text-gray-900 mb-2">Missing Requirements:</h4>
                          <div className="flex flex-wrap gap-1">
                            {match.missing_requirements.map((req) => (
                              <span key={req} className="px-2 py-1 bg-red-100 text-red-800 rounded text-xs">
                                {req}
                              </span>
                            ))}
                          </div>
                        </div>
                      </div>

                      {match.evidence_snippets.length > 0 && (
                        <div className="mt-4">
                          <h4 className="text-sm font-medium text-gray-900 mb-2">Evidence:</h4>
                          <div className="space-y-1">
                            {match.evidence_snippets.slice(0, 2).map((snippet, index) => (
                              <p key={index} className="text-sm text-gray-600 bg-gray-50 p-2 rounded">
                                "{snippet.trim()}"
                              </p>
                            ))}
                          </div>
                        </div>
                      )}
                    </div>

                    <div className="ml-4 flex flex-col items-end space-y-2">
                      <button
                        onClick={() => window.open(`/candidates/${match.resume_id}`, '_blank')}
                        className="btn-primary text-sm"
                      >
                        View Resume
                      </button>
                      <span className="text-xs text-gray-500">
                        Similarity: {(match.similarity_score * 100).toFixed(1)}%
                      </span>
                    </div>
                  </div>
                </div>
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  )
}

export default JobsPage
