import React, { useState } from 'react';
import {
  Button,
  ButtonSet,
  FileUploaderDropContainer,
  FileUploaderItem,
} from '@carbon/react';

import { Upload, SubtractAlt } from '@carbon/icons-react';

interface UploadedFile {
  id: string;
  name: string;
  status: 'edit' | 'complete' | 'uploading';
  invalid?: boolean;
  errorMessage?: string;
}

interface DocumentUploaderProps {
  maxFileSize?: number; // in bytes
  onFilesSelected?: (files: File[]) => void;
}

const UploadFileComponent: React.FC<DocumentUploaderProps> = ({
  maxFileSize = 5 * 1024 * 1024, // 5MB default
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
      <ButtonSet stacked>
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
