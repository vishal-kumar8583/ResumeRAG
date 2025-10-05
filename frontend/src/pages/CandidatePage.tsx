import React from 'react'
import { useParams } from 'react-router-dom'
import { useQuery } from 'react-query'
import axios from 'axios'
import {
  UserIcon,
  DocumentTextIcon,
  CalendarIcon,
  BuildingOfficeIcon,
  AcademicCapIcon,
  SparklesIcon,
  ArrowLeftIcon,
} from '@heroicons/react/24/outline'

const CandidatePage: React.FC = () => {
  const { id } = useParams<{ id: string }>()

  const { data: resume, isLoading, error } = useQuery(
    ['resume', id],
    async () => {
      const response = await axios.get(`/api/resumes/${id}`)
      return response.data
    },
    {
      enabled: !!id,
    }
  )

  if (isLoading) {
    return (
      <div className="flex items-center justify-center min-h-96">
        <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-primary-500"></div>
      </div>
    )
  }

  if (error || !resume) {
    return (
      <div className="text-center py-12">
        <DocumentTextIcon className="mx-auto h-12 w-12 text-gray-400" />
        <h3 className="mt-4 text-lg font-medium text-gray-900">Resume not found</h3>
        <p className="mt-2 text-gray-500">
          The requested resume could not be found or you don't have permission to view it.
        </p>
        <button
          onClick={() => window.history.back()}
          className="mt-4 btn-primary"
        >
          Go Back
        </button>
      </div>
    )
  }

  const extractNameFromContent = (content: string) => {
    // Simple name extraction from the first few lines
    const lines = content.split('\n').filter(line => line.trim()).slice(0, 5)
    for (const line of lines) {
      const words = line.trim().split(/\s+/)
      if (words.length >= 2 && words.length <= 4) {
        if (words.every(word => word.match(/^[A-Z][a-z]+$/))) {
          return line.trim()
        }
      }
    }
    return 'Candidate'
  }

  const formatExperience = (experience: any[]) => {
    if (!experience || experience.length === 0) return []
    return experience.map(exp => ({
      period: exp.period || 'Unknown Period',
      description: exp.context || 'No description available'
    }))
  }

  const candidateName = extractNameFromContent(resume.content)
  const formattedExperience = formatExperience(resume.experience)

  return (
    <div className="space-y-8">
      {/* Header */}
      <div className="flex items-center space-x-4">
        <button
          onClick={() => window.history.back()}
          className="btn-secondary"
        >
          <ArrowLeftIcon className="h-5 w-5 mr-2" />
          Back
        </button>
        <div className="flex-1">
          <h1 className="text-3xl font-bold text-gray-900">Candidate Profile</h1>
          <p className="mt-2 text-gray-600">
            Detailed view of resume and extracted information
          </p>
        </div>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-8">
        {/* Left Column - Candidate Info */}
        <div className="lg:col-span-1 space-y-6">
          {/* Profile Card */}
          <div className="card">
            <div className="p-6">
              <div className="text-center">
                <div className="mx-auto h-24 w-24 rounded-full bg-primary-100 flex items-center justify-center">
                  <UserIcon className="h-12 w-12 text-primary-600" />
                </div>
                <h2 className="mt-4 text-xl font-bold text-gray-900">{candidateName}</h2>
                <p className="text-gray-600">Resume ID: {resume.id}</p>
              </div>

              <div className="mt-6 space-y-3">
                <div className="flex items-center text-sm text-gray-600">
                  <DocumentTextIcon className="h-5 w-5 mr-3 text-gray-400" />
                  <span>{resume.filename}</span>
                </div>
                <div className="flex items-center text-sm text-gray-600">
                  <CalendarIcon className="h-5 w-5 mr-3 text-gray-400" />
                  <span>Uploaded {new Date(resume.created_at).toLocaleDateString()}</span>
                </div>
                <div className="flex items-center text-sm text-gray-600">
                  <SparklesIcon className="h-5 w-5 mr-3 text-gray-400" />
                  <span>{resume.word_count} words</span>
                </div>
              </div>

              {resume.is_pii_redacted && (
                <div className="mt-4 p-3 bg-yellow-50 border border-yellow-200 rounded-lg">
                  <p className="text-sm text-yellow-800">
                    <strong>Note:</strong> Personal information has been redacted for privacy.
                  </p>
                </div>
              )}
            </div>
          </div>

          {/* Skills */}
          {resume.skills && resume.skills.length > 0 && (
            <div className="card">
              <div className="p-6">
                <h3 className="text-lg font-semibold text-gray-900 mb-4">
                  Extracted Skills
                </h3>
                <div className="flex flex-wrap gap-2">
                  {resume.skills.map((skill: string) => (
                    <span
                      key={skill}
                      className="px-3 py-1 bg-primary-100 text-primary-800 rounded-full text-sm font-medium"
                    >
                      {skill}
                    </span>
                  ))}
                </div>
              </div>
            </div>
          )}

          {/* Experience */}
          {formattedExperience.length > 0 && (
            <div className="card">
              <div className="p-6">
                <h3 className="text-lg font-semibold text-gray-900 mb-4 flex items-center">
                  <BuildingOfficeIcon className="h-5 w-5 mr-2 text-gray-400" />
                  Work Experience
                </h3>
                <div className="space-y-4">
                  {formattedExperience.map((exp, index) => (
                    <div key={index} className="border-l-2 border-gray-200 pl-4">
                      <div className="text-sm font-medium text-gray-900">
                        {exp.period}
                      </div>
                      <div className="text-sm text-gray-600 mt-1">
                        {exp.description.length > 150
                          ? `${exp.description.substring(0, 150)}...`
                          : exp.description
                        }
                      </div>
                    </div>
                  ))}
                </div>
              </div>
            </div>
          )}
        </div>

        {/* Right Column - Resume Content */}
        <div className="lg:col-span-2">
          <div className="card">
            <div className="px-6 py-4 border-b border-gray-200">
              <h3 className="text-lg font-semibold text-gray-900">Resume Content</h3>
              <p className="text-sm text-gray-600 mt-1">
                Full text content extracted from the resume
              </p>
            </div>
            <div className="p-6">
              <div className="prose max-w-none">
                <div className="whitespace-pre-wrap text-sm text-gray-700 leading-relaxed">
                  {resume.content}
                </div>
              </div>
            </div>
          </div>
        </div>
      </div>

      {/* Actions */}
      <div className="flex justify-between items-center pt-6 border-t border-gray-200">
        <div className="flex space-x-3">
          <button className="btn-secondary">
            Download Resume
          </button>
          <button className="btn-secondary">
            Add to Favorites
          </button>
        </div>
        <div className="flex space-x-3">
          <button className="btn-primary">
            Contact Candidate
          </button>
          <button className="btn-primary">
            Schedule Interview
          </button>
        </div>
      </div>
    </div>
  )
}

export default CandidatePage
