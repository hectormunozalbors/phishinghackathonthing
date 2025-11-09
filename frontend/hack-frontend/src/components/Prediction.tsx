import React from 'react'

interface Props{
    prediction : string
}

export const Prediction: React.FC<Props> = ({prediction}) => {
  return (
    <div>Prediction: {prediction}</div>
  )
}

export default Prediction
