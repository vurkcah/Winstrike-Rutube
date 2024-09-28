import React from 'react'
import Form from "../../components/Form/Form.jsx";
import {Link} from "react-router-dom";

import './NotFound.scss'

export const NotFound = () => {
    return (
        <div className='container'>
            <div className='main-box'>
                <div className='notFound'>
                    <p>Cтраница не найдена</p>
                    <Link to='/'>
                        <button>Вернуться на главную</button>
                    </Link>
                </div>
            </div>
        </div>
    )
}