//******************* An example MySQL config *******************//

// We can also create a s3 bucket to store the docs that are uploaded to the vectordb
// which we could send as reference back from the API and then provide to the end user 
// to look at the original doc as source.

// will implement with fileuploader at a later date

// import mysql from 'mysql2/promise';
// import dotenv from 'dotenv'

// dotenv.config()

// const DB_HOST = process.env.DB_HOST || '';
// const DB_NAME = process.env.DB_NAME || '';
// const DB_USER = process.env.DB_USER || '';
// const DB_PASSWORD = process.env.DB_PASSWORD || '';
// const DB_PORT = Number(process.env.DB_PORT) || 3306;

// console.log(`In dbConfig: ${DB_PORT}, ${DB_PASSWORD} ${DB_NAME} ${DB_HOST}`)


// export const createDBConnection = async () => {
//   return mysql.createConnection({
//     host: DB_HOST,
//     user: DB_USER,
//     port: DB_PORT,
//     password: DB_PASSWORD,
//     database: DB_NAME
//   });
// }
