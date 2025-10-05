import React, { useState } from 'react'
import { useDropzone } from 'react-dropzone'
import { useMutation, useQueryClient } from 'react-query'
import axios from 'axios'
import toast from 'react-hot-toast'
import {
  CloudArrowUpIcon,
  DocumentTextIcon,
  TrashIcon,
  CheckCircleIcon,
  ExclamationTriangleIcon,
} from '@heroicons/react/24/outline'

interface UploadedFile {
  file: File
  status: 'pending' | 'uploading' | 'success' | 'error'
  result?: any
  error?: string
}

const UploadPage: React.FC = () => {
  const [files, setFiles] = useState<UploadedFile[]>([])
  const queryClient = useQueryClient()

  const uploadMutation = useMutation(
    async (filesToUpload: File[]) => {
      const formData = new FormData()
      filesToUpload.forEach((file) => {
        formData.append('files', file)
      })

      // Add idempotency key
      const idempotencyKey = `upload_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`
      
      const response = await axios.post('/api/resumes', formData, {
        headers: {
          'Content-Type': 'multipart/form-data',
          'Idempotency-Key': idempotencyKey,
        },
      })
      return response.data
    },
    {
      onSuccess: (data) => {
        // Update file statuses
        setFiles((prev) => 
          prev.map((f) => 
            f.status === 'uploading' 
              ? { ...f, status: 'success', result: data }
              : f
          )
        )
        toast.success(`Successfully uploaded ${data.count} resume(s)!`)
        queryClient.invalidateQueries('dashboard-stats')
        queryClient.invalidateQueries('resumes')
      },
      onError: (error: any) => {
        console.error('Upload error:', error)
        const errorMessage = error.response?.data?.error?.message || 'Upload failed'
        
        // Update file statuses
        setFiles((prev) => 
          prev.map((f) => 
            f.status === 'uploading' 
              ? { ...f, status: 'error', error: errorMessage }
              : f
          )
        )
        toast.error(errorMessage)
      },
    }
  )

  const onDrop = (acceptedFiles: File[]) => {
    const validFiles = acceptedFiles.filter((file) => {
      const validTypes = ['.pdf', '.docx', '.doc', '.txt', '.zip']
      const fileExtension = '.' + file.name.split('.').pop()?.toLowerCase()
      return validTypes.includes(fileExtension)
    })

    if (validFiles.length !== acceptedFiles.length) {
      toast.error('Some files were rejected. Only PDF, DOCX, DOC, TXT, and ZIP files are allowed.')
    }

    const newFiles: UploadedFile[] = validFiles.map((file) => ({
      file,
      status: 'pending'
    }))

    setFiles((prev) => [...prev, ...newFiles])
  }

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop,
    accept: {
      'application/pdf': ['.pdf'],
      'application/vnd.openxmlformats-officedocument.wordprocessingml.document': ['.docx'],
      'application/msword': ['.doc'],
      'text/plain': ['.txt'],
      'application/zip': ['.zip'],
    },
    multiple: true,
  })

  const handleUpload = () => {
    const pendingFiles = files.filter((f) => f.status === 'pending')
    if (pendingFiles.length === 0) {
      toast.error('No files to upload')
      return
    }

    // Update status to uploading
    setFiles((prev) => 
      prev.map((f) => 
        f.status === 'pending' 
          ? { ...f, status: 'uploading' }
          : f
      )
    )

    uploadMutation.mutate(pendingFiles.map((f) => f.file))
  }

  const removeFile = (index: number) => {
    setFiles((prev) => prev.filter((_, i) => i !== index))
  }

  const clearAll = () => {
    setFiles([])
  }

  const getFileIcon = (fileName: string) => {
    const extension = fileName.split('.').pop()?.toLowerCase()
    return <DocumentTextIcon className="h-8 w-8 text-gray-400" />
  }

  const getStatusIcon = (status: string) => {
    switch (status) {
      case 'success':
        return <CheckCircleIcon className="h-5 w-5 text-green-500" />
      case 'error':
        return <ExclamationTriangleIcon className="h-5 w-5 text-red-500" />
      case 'uploading':
        return <div className="animate-spin rounded-full h-5 w-5 border-b-2 border-primary-500"></div>
      default:
        return null
    }
  }

  const formatFileSize = (bytes: number) => {
    if (bytes === 0) return '0 Bytes'
    const k = 1024
    const sizes = ['Bytes', 'KB', 'MB', 'GB']
    const i = Math.floor(Math.log(bytes) / Math.log(k))
    return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i]
  }

  return (
    <div className="space-y-8">
      <div>
        <h1 className="text-3xl font-bold text-gray-900">Upload Resumes</h1>
        <p className="mt-2 text-gray-600">
          Upload PDF, DOCX, DOC, TXT files, or ZIP archives containing multiple resumes.
          Our AI will automatically extract and index the content.
        </p>
      </div>

      {/* Upload Area */}
      <div className="card">
        <div className="p-6">
          <div
            {...getRootProps()}
            className={`border-2 border-dashed rounded-lg p-8 text-center cursor-pointer transition-colors duration-200 ${
              isDragActive
                ? 'border-primary-500 bg-primary-50'
                : 'border-gray-300 hover:border-gray-400'
            }`}
          >
            <input {...getInputProps()} />
            <CloudArrowUpIcon className="mx-auto h-12 w-12 text-gray-400" />
            <h3 className="mt-4 text-lg font-medium text-gray-900">
              {isDragActive ? 'Drop files here' : 'Drop files here, or click to browse'}
            </h3>
            <p className="mt-2 text-sm text-gray-500">
              PDF, DOCX, DOC, TXT, ZIP files up to 10MB each
            </p>
          </div>
        </div>
      </div>

      {/* File List */}
      {files.length > 0 && (
        <div className="card">
          <div className="px-6 py-4 border-b border-gray-200 flex items-center justify-between">
            <h3 className="text-lg font-medium text-gray-900">
              Files to Upload ({files.length})
            </h3>
            <div className="flex space-x-3">
              {files.some((f) => f.status === 'pending') && (
                <button
                  onClick={handleUpload}
                  disabled={uploadMutation.isLoading}
                  className="btn-primary disabled:opacity-50 disabled:cursor-not-allowed"
                >
                  {uploadMutation.isLoading ? 'Uploading...' : 'Upload All'}
                </button>
              )}
              <button
                onClick={clearAll}
                className="btn-secondary"
              >
                Clear All
              </button>
            </div>
          </div>
          <div className="divide-y divide-gray-200">
            {files.map((fileItem, index) => (
              <div key={index} className="px-6 py-4 flex items-center space-x-4">
                <div className="flex-shrink-0">
                  {getFileIcon(fileItem.file.name)}
                </div>
                <div className="flex-1 min-w-0">
                  <p className="text-sm font-medium text-gray-900 truncate">
                    {fileItem.file.name}
                  </p>
                  <p className="text-sm text-gray-500">
                    {formatFileSize(fileItem.file.size)}
                  </p>
                  {fileItem.error && (
                    <p className="text-sm text-red-600 mt-1">
                      {fileItem.error}
                    </p>
                  )}
                  {fileItem.result && (
                    <p className="text-sm text-green-600 mt-1">
                      Successfully processed
                    </p>
                  )}
                </div>
                <div className="flex items-center space-x-3">
                  {getStatusIcon(fileItem.status)}
                  <button
                    onClick={() => removeFile(index)}
                    className="text-gray-400 hover:text-red-500 transition-colors duration-200"
                  >
                    <TrashIcon className="h-5 w-5" />
                  </button>
                </div>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Tips */}
      <div className="bg-blue-50 border border-blue-200 rounded-lg p-6">
        <h3 className="text-lg font-medium text-blue-900 mb-4">Upload Tips</h3>
        <ul className="space-y-2 text-sm text-blue-800">
          <li className="flex items-start">
            <span className="font-medium mr-2">•</span>
            <span>ZIP files will be automatically extracted and all resumes inside will be processed</span>
          </li>
          <li className="flex items-start">
            <span className="font-medium mr-2">•</span>
            <span>Personal information (PII) is automatically redacted for non-recruiter users</span>
          </li>
          <li className="flex items-start">
            <span className="font-medium mr-2">•</span>
            <span>Our AI extracts skills, experience, and creates searchable embeddings</span>
          </li>
          <li className="flex items-start">
            <span className="font-medium mr-2">•</span>
            <span>All uploads are idempotent - duplicate uploads will be handled gracefully</span>
          </li>
        </ul>
      </div>
    </div>
  )
}

export default UploadPage
