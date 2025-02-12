import express from 'express';
import configRouter from './configRoutes';
import dbRouter from './dbRoutes';
import queryRouter from './queryRoutes'

const router = express.Router();

router.use('/api', configRouter);
router.use('/api/db', dbRouter);
router.use('/api/query', queryRouter)

export default router;
