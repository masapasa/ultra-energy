// pages/index.tsx
import { GetStaticProps } from 'next'
import { useState } from 'react'
type SensorData = {
  id: number
  timestamp: string
  temperature: number
  humidity: number
}
type Props = {
  sensorData: SensorData[]
}
export const getStaticProps: GetStaticProps<Props> = async () => {
    const res = await fetch(
      'https://raw.githubusercontent.com/masapasa/ultra-energy/master/data/sensor-data.json'
    )
    const sensorData: SensorData[] = await res.json()
    return {
      props: {
        sensorData,
      },
    }
  }
export default function Home({ sensorData }: Props) {
  const [data, setData] = useState(sensorData)
  
  return (
    <div>
      <h1>Sensor Data</h1>
      <table>
        <thead>
          <tr>
            <th>ID</th>
            <th>Timestamp</th>
            <th>Temperature</th>
            <th>Humidity</th>
          </tr>
        </thead>
        <tbody>
          {data.map((d) => (
            <tr key={d.id}>
              <td>{d.id}</td>
              <td>{d.timestamp}</td>
              <td>{d.temperature}</td>
              <td>{d.humidity}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  )
}
