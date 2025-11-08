import React, { useState } from 'react'
import "./SearchBar.css"
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
    const handleValueChange = (e: React.ChangeEvent<HTMLTextAreaElement>) =>{
        setValue(e.target.value);
    }
    return (
        <div>
            {/* 💡 CHANGE 1: Change the tag from <input> to <textarea> */}
            <textarea 
            placeholder='Enter Email...'
            value={value}
            onChange={handleValueChange}
            rows={16} 
            cols={80} 
            />
            <button onClick={handleClick}>
                Submit
            </button>
        </div>
        
    )
}

export default SearchBar