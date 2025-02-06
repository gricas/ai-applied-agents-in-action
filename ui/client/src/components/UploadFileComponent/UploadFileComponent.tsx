import React, { useState } from 'react';
import {
  Button,
  ButtonSet,
  FileUploaderDropContainer,
  FileUploaderItem,
} from '@carbon/react';
import axios from 'axios';

import { Upload, SubtractAlt } from '@carbon/icons-react';

interface UploadedFile {
  id: string;
  name: string;
  status: 'edit' | 'complete' | 'uploading';
  invalid?: boolean;
  errorMessage?: string;
  file?: File;
}

interface DocumentUploaderProps {
  maxFileSize?: number;
  onFilesSelected?: (files: File[]) => void;
}

const UploadFileComponent: React.FC<DocumentUploaderProps> = ({
  maxFileSize = 5 * 1024 * 1024,
  onFilesSelected,
}) => {
  const [uploadedFiles, setUploadedFiles] = useState<UploadedFile[]>([]);

  const acceptedTypes = [
    'text/plain',
    'text/markdown',
    'application/pdf',
    '.txt',
    '.md',
    '.pdf',
  ];

  const generateFileId = () =>
    `file-${Math.random().toString(36).substr(2, 9)}`;

  const validateFile = (file: File): { valid: boolean; error?: string } => {
    if (
      !acceptedTypes.includes(file.type) &&
      !acceptedTypes.some((type) => file.name.toLowerCase().endsWith(type))
    ) {
      return {
        valid: false,
        error:
          'File type not supported. Please upload .txt, .md, or .pdf files.',
      };
    }

    if (file.size > maxFileSize) {
      return {
        valid: false,
        error: `File size exceeds ${maxFileSize / (1024 * 1024)}MB limit.`,
      };
    }

    return { valid: true };
  };

  const handleFilesAdded = (event: React.ChangeEvent<HTMLInputElement>) => {
    const files = event.target.files;
    if (!files) return;

    const newFiles: UploadedFile[] = [];
    const validFiles: File[] = [];

    Array.from(files).forEach((file) => {
      const validation = validateFile(file);
      const fileData: UploadedFile = {
        id: generateFileId(),
        name: file.name,
        status: 'edit',
        invalid: !validation.valid,
        errorMessage: validation.error,
        file: file,
      };

      if (validation.valid) {
        validFiles.push(file);
      }

      newFiles.push(fileData);
    });

    setUploadedFiles((prev) => [...prev, ...newFiles]);

    if (validFiles.length > 0 && onFilesSelected) {
      onFilesSelected(validFiles);
    }
  };

  const handleDrop = (
    event: React.SyntheticEvent<HTMLElement, Event>,
    { addedFiles }: { addedFiles: File[] },
  ) => {
    const filesArray = addedFiles;
    const event2 = {
      target: { files: filesArray as unknown as FileList },
    } as React.ChangeEvent<HTMLInputElement>;
    handleFilesAdded(event2);
  };

  const handleDelete = (fileId: string) => {
    setUploadedFiles((prev) => prev.filter((file) => file.id !== fileId));
  };

  const handleSubmit = async () => {
    // Filter out invalid files
    const validFiles = uploadedFiles.filter(
      (file) => !file.invalid && file.file,
    );

    if (validFiles.length === 0) {
      console.error('No valid files to upload');
      return;
    }

    // Create FormData instance
    const formData = new FormData();

    // Update UI to show uploading status
    setUploadedFiles((prev) =>
      prev.map((file) => ({
        ...file,
        status: !file.invalid ? 'uploading' : file.status,
      })),
    );

    // Append all valid files to FormData
    validFiles.forEach((fileData) => {
      if (fileData.file) {
        formData.append('files', fileData.file);
      }
    });

    try {
      const response = await axios.post('/api/uploadFile', formData, {
        headers: {
          'Content-Type': 'multipart/form-data',
        },
        onUploadProgress: (progressEvent) => {
          const percentCompleted = Math.round(
            (progressEvent.loaded * 100) / (progressEvent.total || 100),
          );
          console.log(`Upload Progress: ${percentCompleted}%`);
        },
      });

      // Update UI to show completion status
      setUploadedFiles((prev) =>
        prev.map((file) => ({
          ...file,
          status: !file.invalid ? 'complete' : file.status,
        })),
      );

      console.log('Upload successful:', response.data);
    } catch (error) {
      // Update UI to show error status
      setUploadedFiles((prev) =>
        prev.map((file) => ({
          ...file,
          status: 'edit',
          invalid: true,
          errorMessage: 'Upload failed. Please try again.',
        })),
      );

      console.error('Upload failed:', error);
    }
  };

  return (
    <>
      <p className='cds--file--label'>Upload Documents</p>
      <p className='cds--label-description'>
        Max file size is {maxFileSize / (1024 * 1024)}MB. Supported file types
        are .txt, .md, and .pdf.
      </p>

      <FileUploaderDropContainer
        accept={acceptedTypes}
        labelText='Drag and drop files here or click to upload'
        multiple={true}
        onAddFiles={handleDrop}
        name='document-upload'
      />

      <div className='cds--file-container'>
        {uploadedFiles.map((file) => (
          <FileUploaderItem
            key={file.id}
            uuid={file.id}
            name={file.name}
            status={file.status}
            invalid={file.invalid}
            errorBody={file.errorMessage}
            errorSubject='Upload Error'
            iconDescription='Remove file'
            onDelete={() => handleDelete(file.id)}
            size='lg'
          />
        ))}
      </div>
      <ButtonSet stacked className='upload-file__button-set'>
        <Button
          kind='primary'
          renderIcon={Upload}
          iconDescription='Upload'
          onClick={() => console.log(uploadedFiles)}
        >
          Upload
        </Button>
        <Button
          kind='secondary'
          renderIcon={SubtractAlt}
          iconDescription='Clear List'
          onClick={() => setUploadedFiles([])}
        >
          Clear
        </Button>
      </ButtonSet>
    </>
  );
};

export default UploadFileComponent;
