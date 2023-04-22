import type { NextPage } from 'next'
import Head from 'next/head'
import Image from 'next/image'
import styles from '../styles/Home.module.css'
import { useState, useEffect } from 'react'

type SensorData = {
  id: number,
  temp: number,
  humidity: number,
  pressure: number
}

const Home: NextPage = () => {
  const [data, setData] = useState<SensorData[]>([]);

  useEffect(() => {
    async function fetchData() {
      const res = await fetch('data/sensor-data.json');
      const json = await res.json();
      setData(json);
    }

    fetchData();
  }, [])

  return (
    <div className={styles.container}>
      <Head>
        <title>Sensor Data</title>
        <meta name="description" content="Display sensor data" />
        <link rel="icon" href="/favicon.ico" />
      </Head>

      <main className={styles.main}>
        <h1 className={styles.title}>
          Sensor Data
        </h1>

        <div className={styles.grid}>
          {data.map((item) => (
            <a key={item.id} className={styles.card}>
              <h2>Temperature: {item.temp}°F</h2>
              <p>Humidity: {item.humidity}%</p>
              <p>Pressure: {item.pressure}Pa</p>
            </a>
          ))}
        </div>
      </main>

      <footer className={styles.footer}>
        <a
          href="https://vercel.com?utm_source=create-next-app&utm_medium=default-template&utm_campaign=create-next-app"
          target="_blank"
          rel="noopener noreferrer"
        >
          Powered by{' '}
          <span className={styles.logo}>
            <Image src="/vercel.svg" alt="Vercel Logo" width={72} height={16} />
          </span>
        </a>
      </footer>
    </div>
  )
}

export default Home