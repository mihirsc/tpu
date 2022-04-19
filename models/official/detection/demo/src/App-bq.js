import React, { useState } from 'react';
import Axios from 'axios';
import './App.css';

function App() {
    
    const [file, setFile] = useState(null)
    const [images, setImages] = useState(null)
    const [imagesMultiple, setMultipleImages] = useState(null)
    
    const handleFiles = (e) => {
        setFile(e.target.files);
    }
    
    const handleSingleUpload = (e) => {
        let formdata = new FormData();
        formdata.append('query', file[0]);
        Axios({
            url: 'http://10.128.0.225:5000/single_image_inference',
            method: 'POST',
            data: formdata
        })
        .then((res) => {
            setImages(
                <div>
                <img src={URL.createObjectURL(file[0])} alt='' style={{'width':'150px', 'border':'1px solid'}}/>
                {res.data.results.map(items => {
                    return <div>
                    <br/> <h2 style={{'color':'white'}}>We recommend the following products for the {items[0]} class:</h2> <br/>
                    {items[1].map( item => {
                        return <div style={{'display':'inline-block'}}>
                        <a key={item[0]} href={item[1]} target="_blank" rel="noreferrer">
                        <img src={item[2]} alt='' style={{'margin':'25px', 'width':'150px', 'border':'1px solid', 'display':'block'}}/>
                        </a>
                        </div>
                    })}
                    </div>
                })}
                </div>
            );
        });
    }
    
    const handleMultipleUpload = (e) => {
        let formdata = new FormData();
        for (let i=0; i<file.length; i++) {
            formdata.append('queries', file[i]);
        }
        Axios({
            url: 'http://10.128.0.225:5000/multi_image_inference',
            method: 'POST',
            data: formdata
        })
        .then((res) => {
            console.log(res)
            setMultipleImages(res.data.results.map( (imageRes, index) => {
                console.log(index)
                console.log(imageRes)
                return <div>
                <img src={URL.createObjectURL(file[index])} alt='' style={{'width':'150px', 'border':'1px solid'}}/>
                {imageRes.map( items => {
                    return <div>
                    <br/> <h2 style={{'color':'white'}}>We recommend the following products for the {items[0]} class:</h2> <br/>
                    {items[1].map(item => {
                        return <div style={{'display':'inline-block'}}>
                        <a key={item[0]} href={item[1]} target="_blank" rel="noreferrer">
                        <img src={item[2]} alt='' style={{'margin':'25px', 'width':'150px', 'border':'1px solid', 'display':'block'}}/>
                        </a>
                        </div>
                    })}
                    </div>
                })} 
                <hr/>
                </div>
            }))
        });
    }

    return (
        <div style={{'display':'flex', 'justifyContent':'center'}}>
            <div style={{'textAlign':'center', 'minHeight': '100vh', 'width':'80vw', 'backgroundColor':'orange'}}>
                <h1 style={{'color':'white'}}>Video Commerce Demo</h1>
                <div>
                    <h2 style={{'color':'white'}}>Add a picture to infer on</h2>
                    <br/><br/>
                    <span style={{'padding':'10px', 'border':'1px solid black', 'borderRadius':'5px', 'backgroundColor': 'lightgrey'}}>
                        <input type='file' name='file' onChange={e => {handleFiles(e)}} />
                        <button type='button' onClick={e => {handleSingleUpload(e)}}>Infer</button>
                    </span>
                    <br/><br/>
                    {images}
                </div>
                <br/><br/>
                <div>
                    <h2 style={{'color':'white'}}>Add some pictures to infer on</h2>
                    <br/><br/>
                    <span style={{'padding':'10px', 'border':'1px solid black', 'borderRadius':'5px', 'backgroundColor': 'lightgrey'}}>
                        <input type='file' multiple name='file' onChange={e => {handleFiles(e)}} />
                        <button type='button' onClick={e => {handleMultipleUpload(e)}}>Infer</button>
                    </span>
                    <br/><br/>
                    {imagesMultiple}
                </div>
            </div>
        </div>
    );
}

export default App;