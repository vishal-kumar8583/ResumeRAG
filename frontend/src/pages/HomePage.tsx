import React from 'react'
import { Link } from 'react-router-dom'
import { useQuery } from 'react-query'
import axios from 'axios'
import { useAuth } from '../contexts/AuthContext'
import {
  CloudArrowUpIcon,
  MagnifyingGlassIcon,
  BriefcaseIcon,
  DocumentTextIcon,
  ChartBarIcon,
  SparklesIcon,
} from '@heroicons/react/24/outline'

const HomePage: React.FC = () => {
  const { user } = useAuth()

  const { data: stats } = useQuery('dashboard-stats', async () => {
    const response = await axios.get('/api/resumes?limit=1&offset=0')
    return {
      totalResumes: response.data.total || 0,
      recentUploads: response.data.items || []
    }
  })

  const quickActions = [
    {
      title: 'Upload Resumes',
      description: 'Upload PDF, DOCX, or ZIP files containing resumes',
      icon: CloudArrowUpIcon,
      href: '/upload',
      color: 'bg-primary-500',
      bgColor: 'bg-primary-50',
      textColor: 'text-primary-700'
    },
    {
      title: 'Search Resumes',
      description: 'Ask questions and find relevant resume content',
      icon: MagnifyingGlassIcon,
      href: '/search',
      color: 'bg-green-500',
      bgColor: 'bg-green-50',
      textColor: 'text-green-700'
    },
    {
      title: 'Manage Jobs',
      description: 'Create job postings and match with candidates',
      icon: BriefcaseIcon,
      href: '/jobs',
      color: 'bg-purple-500',
      bgColor: 'bg-purple-50',
      textColor: 'text-purple-700'
    }
  ]

  return (
    <div className="space-y-8">
      {/* Header */}
      <div className="bg-gradient-to-r from-primary-600 to-primary-700 rounded-xl p-8 text-white">
        <div className="flex items-center">
          <SparklesIcon className="h-12 w-12 text-primary-200" />
          <div className="ml-4">
            <h1 className="text-3xl font-bold">
              Welcome back, {user?.email?.split('@')[0]}!
            </h1>
            <p className="text-primary-200 mt-2">
              {user?.role === 'recruiter' ? 
                'Manage job postings and find the perfect candidates for your roles.' :
                'Upload your resumes and discover opportunities with AI-powered search.'
              }
            </p>
          </div>
        </div>
      </div>

      {/* Stats Cards */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
        <div className="card p-6">
          <div className="flex items-center">
            <div className="flex-shrink-0">
              <DocumentTextIcon className="h-8 w-8 text-primary-600" />
            </div>
            <div className="ml-5">
              <p className="text-sm font-medium text-gray-500 uppercase tracking-wide">
                Total Resumes
              </p>
              <p className="text-2xl font-semibold text-gray-900">
                {stats?.totalResumes || 0}
              </p>
            </div>
          </div>
        </div>

        <div className="card p-6">
          <div className="flex items-center">
            <div className="flex-shrink-0">
              <ChartBarIcon className="h-8 w-8 text-green-600" />
            </div>
            <div className="ml-5">
              <p className="text-sm font-medium text-gray-500 uppercase tracking-wide">
                Account Type
              </p>
              <p className="text-2xl font-semibold text-gray-900 capitalize">
                {user?.role}
              </p>
            </div>
          </div>
        </div>

        <div className="card p-6">
          <div className="flex items-center">
            <div className="flex-shrink-0">
              <SparklesIcon className="h-8 w-8 text-purple-600" />
            </div>
            <div className="ml-5">
              <p className="text-sm font-medium text-gray-500 uppercase tracking-wide">
                AI-Powered
              </p>
              <p className="text-2xl font-semibold text-gray-900">
                Ready
              </p>
            </div>
          </div>
        </div>
      </div>

      {/* Quick Actions */}
      <div className="space-y-6">
        <h2 className="text-2xl font-bold text-gray-900">Quick Actions</h2>
        <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
          {quickActions.map((action) => (
            <Link
              key={action.title}
              to={action.href}
              className={`${action.bgColor} rounded-xl p-6 hover:shadow-md transition-all duration-200 group`}
            >
              <div className="flex items-center">
                <div className={`${action.color} rounded-lg p-3`}>
                  <action.icon className="h-6 w-6 text-white" />
                </div>
                <div className="ml-4 flex-1">
                  <h3 className={`text-lg font-semibold ${action.textColor}`}>
                    {action.title}
                  </h3>
                  <p className="text-gray-600 text-sm mt-1">
                    {action.description}
                  </p>
                </div>
              </div>
            </Link>
          ))}
        </div>
      </div>

      {/* Recent Activity */}
      {stats?.recentUploads?.length > 0 && (
        <div className="space-y-6">
          <h2 className="text-2xl font-bold text-gray-900">Recent Activity</h2>
          <div className="card">
            <div className="px-4 py-5 sm:p-6">
              <div className="space-y-4">
                {stats.recentUploads.map((resume: any) => (
                  <div key={resume.id} className="flex items-center space-x-3">
                    <DocumentTextIcon className="h-8 w-8 text-gray-400" />
                    <div className="flex-1">
                      <p className="text-sm font-medium text-gray-900">
                        {resume.filename}
                      </p>
                      <p className="text-sm text-gray-500">
                        {new Date(resume.created_at).toLocaleDateString()} • {resume.word_count} words
                      </p>
                    </div>
                  </div>
                ))}
              </div>
              <div className="mt-6">
                <Link
                  to="/search"
                  className="text-sm font-medium text-primary-600 hover:text-primary-500"
                >
                  View all resumes →
                </Link>
              </div>
            </div>
          </div>
        </div>
      )}

      {/* Getting Started */}
      {(!stats || stats.totalResumes === 0) && (
        <div className="bg-gradient-to-br from-gray-50 to-gray-100 rounded-xl p-8 border border-gray-200">
          <div className="text-center">
            <CloudArrowUpIcon className="mx-auto h-12 w-12 text-gray-400" />
            <h3 className="mt-4 text-lg font-semibold text-gray-900">
              Get Started with ResumeRAG
            </h3>
            <p className="mt-2 text-gray-600 max-w-2xl mx-auto">
              Upload your first resume to experience the power of AI-driven resume search and matching.
              Our system will automatically extract skills, experience, and key information.
            </p>
            <div className="mt-6">
              <Link
                to="/upload"
                className="btn-primary"
              >
                Upload Your First Resume
              </Link>
            </div>
          </div>
        </div>
      )}
    </div>
  )
}

export default HomePage
