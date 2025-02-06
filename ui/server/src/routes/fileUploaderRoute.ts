import express from 'express';
import multer from 'multer';
import path from 'path';
import { Request, Response } from 'express';

// Custom type for file upload request
interface FileUploadRequest extends Request {
  files?: Express.Multer.File[] | { [fieldname: string]: Express.Multer.File[] };
}

// Configure multer storage
const storage = multer.diskStorage({
  destination: (req, file, cb) => {
    cb(null, 'uploads/'); // Make sure this directory exists
  },
  filename: (req, file, cb) => {
    // Create unique filename with original extension
    const uniqueSuffix = `${Date.now()}-${Math.round(Math.random() * 1E9)}`;
    cb(null, `${file.fieldname}-${uniqueSuffix}${path.extname(file.originalname)}`);
  }
});

// Configure file filter
const fileFilter = (req: Request, file: Express.Multer.File, cb: multer.FileFilterCallback) => {
  const allowedTypes = ['text/plain', 'text/markdown', 'application/pdf'];
  const allowedExtensions = ['.txt', '.md', '.pdf'];
  
  const isValidMimeType = allowedTypes.includes(file.mimetype);
  const isValidExtension = allowedExtensions.some(ext => 
    file.originalname.toLowerCase().endsWith(ext)
  );

  if (isValidMimeType || isValidExtension) {
    cb(null, true);
  } else {
    cb(new Error('Invalid file type. Only .txt, .md, and .pdf files are allowed.'));
  }
};

// Configure multer
const upload = multer({
  storage: storage,
  fileFilter: fileFilter,
  limits: {
    fileSize: 5 * 1024 * 1024, // 5MB
    files: 10 // Maximum number of files
  }
});

// Create router
const router = express.Router();

// Upload endpoint
router.post('/upload', upload.array('documents'), async (req: Request, res: Response) => {
  try {
    const files = req.files as Express.Multer.File[] | undefined;
    // Ensure files were uploaded
    if (!files || files.length === 0) {
      return res.status(400).json({
        success: false,
        message: 'No files were uploaded.'
      });
    }

    // Process uploaded files
    const uploadedFiles = files.map(file => ({
      filename: file.filename,
      originalName: file.originalname,
      size: file.size,
      mimetype: file.mimetype,
      path: file.path
    }));

    // Here you would typically:
    // 1. Save file metadata to database
    // 2. Process files as needed
    // 3. Forward files to FastAPI service

    res.status(200).json({
      success: true,
      message: 'Files uploaded successfully',
      files: uploadedFiles
    });

  } catch (error) {
    console.error('File upload error:', error);
    res.status(500).json({
      success: false,
      message: 'Error processing file upload',
      error: error instanceof Error ? error.message : 'Unknown error'
    });
  }
});

// Error handling middleware
router.use((error: Error, req: Request, res: Response, next: express.NextFunction) => {
  if (error instanceof multer.MulterError) {
    if (error.code === 'LIMIT_FILE_SIZE') {
      return res.status(400).json({
        success: false,
        message: 'File size exceeds limit of 5MB.'
      });
    }
    if (error.code === 'LIMIT_FILE_COUNT') {
      return res.status(400).json({
        success: false,
        message: 'Too many files. Maximum is 10 files.'
      });
    }
  }
  
  res.status(500).json({
    success: false,
    message: error.message || 'Something went wrong during file upload.'
  });
});

export default router;
