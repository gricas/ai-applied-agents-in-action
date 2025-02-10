import express, { Request, Response } from 'express';
import axios from 'axios';
import dotenv from 'dotenv';

dotenv.config();

const router = express.Router();

const API_URL: string = process.env.API_URL || '';

router.get('/health', (_req: Request, res: Response) => {
  res.status(200).send({ message: 'Query route up and working!' });
});

router.post('/', async (req: Request, res: Response) => {
  try {
    const { query } = req.body;

    if (!query) {
      return res.status(400).send({ error: 'Query is required' });
    }

    // const result = await axios.post(`${API_URL}/rag-query`, { query });
    const result = await axios.post(`${API_URL}/agentic-route`, { query });

    const answer = result.data.response.json_dict.response;
    const category = result.data.response.json_dict.category;

    console.log(`Answer: ${answer}, Category: ${category}`)

    return res.status(200).send({
      answer,
      category
    });
  } catch (error) {
    if (axios.isAxiosError(error)) {
      console.error('API Error:', error.response?.data || error.message);
      return res.status(error.response?.status || 500).send({
        error: error.response?.data || 'Error calling RAG query API',
      });
    }
    console.error('Server Error:', error);
    return res.status(500).send({
      error: 'Internal server error',
    });
  }
});

export default router;
