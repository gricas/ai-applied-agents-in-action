import express from 'express';
import multer from 'multer';
import path from 'path';
import { Request, Response } from 'express';


// the route to upload multiple files, will complete a bit later to give the process-documents route in 
// the api access to docling, so we can upload any type of doc, which will be sweet.

const router = express.Router();

interface FileUploadRequest extends Request {
  files?:
    | Express.Multer.File[]
    | { [fieldname: string]: Express.Multer.File[] };
}

const storage = multer.diskStorage({
  destination: (req, file, cb) => {
    cb(null, 'uploads/');
  },
  filename: (req, file, cb) => {
    const uniqueSuffix = `${Date.now()}-${Math.round(Math.random() * 1e9)}`;
    cb(
      null,
      `${file.fieldname}-${uniqueSuffix}${path.extname(file.originalname)}`,
    );
  },
});

const fileFilter = (
  req: Request,
  file: Express.Multer.File,
  cb: multer.FileFilterCallback,
) => {
  const allowedTypes = ['text/plain', 'text/markdown', 'application/pdf'];
  const allowedExtensions = ['.txt', '.md', '.pdf'];

  const isValidMimeType = allowedTypes.includes(file.mimetype);
  const isValidExtension = allowedExtensions.some((ext) =>
    file.originalname.toLowerCase().endsWith(ext),
  );

  if (isValidMimeType || isValidExtension) {
    cb(null, true);
  } else {
    cb(
      new Error(
        'Invalid file type. Only .txt, .md, and .pdf files are allowed.',
      ),
    );
  }
};

const upload = multer({
  storage: storage,
  fileFilter: fileFilter,
  limits: {
    fileSize: 5 * 1024 * 1024,
    files: 10,
  },
});


router.post(
  '/upload',
  upload.array('documents'),
  async (req: Request, res: Response) => {
    try {
      const files = req.files as Express.Multer.File[] | undefined;
      if (!files || files.length === 0) {
        return res.status(400).json({
          success: false,
          message: 'No files were uploaded.',
        });
      }

      const uploadedFiles = files.map((file) => ({
        filename: file.filename,
        originalName: file.originalname,
        size: file.size,
        mimetype: file.mimetype,
        path: file.path,
      }));

      res.status(200).json({
        success: true,
        message: 'Files uploaded successfully',
        files: uploadedFiles,
      });
    } catch (error) {
      console.error('File upload error:', error);
      res.status(500).json({
        success: false,
        message: 'Error processing file upload',
        error: error instanceof Error ? error.message : 'Unknown error',
      });
    }
  },
);

router.use(
  (error: Error, req: Request, res: Response, next: express.NextFunction) => {
    if (error instanceof multer.MulterError) {
      if (error.code === 'LIMIT_FILE_SIZE') {
        return res.status(400).json({
          success: false,
          message: 'File size exceeds limit of 5MB.',
        });
      }
      if (error.code === 'LIMIT_FILE_COUNT') {
        return res.status(400).json({
          success: false,
          message: 'Too many files. Maximum is 10 files.',
        });
      }
    }

    res.status(500).json({
      success: false,
      message: error.message || 'Something went wrong during file upload.',
    });
  },
);

export default router;
