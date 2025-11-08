import React, { useState } from 'react'
import App from '../App';

interface Props{
    email : string
    setEmail : (e : string) => void;
}

const SearchBar: React.FC<Props> = ({email,setEmail}) => {
    const [value,setValue] = useState("");
    const handleClick = () =>{
        setEmail(value);
        console.log(email)
    }
    const handleValueChange = (e : React.ChangeEvent<HTMLInputElement>) =>{
        setValue(e.target.value);
    }
    return (
        <div>
            <input 
            type = "text"
            placeholder='Enter Email...'
            value = {value}
            onChange={handleValueChange}
            />
            <button onClick={handleClick}>
                Submit
            </button>
        </div>
        
    )
}

export default SearchBar