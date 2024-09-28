import React from 'react'
import style from './Header.module.scss'
import { Link, useLocation } from 'react-router-dom'

import logo from '../../assets/rutube.png.png'

export const Header = () => {
	const location = useLocation()

	return (
		<div className={style.header}>
			<img className={style.header__logo} src={logo} alt='rutube-logo' />
			<nav className={style.header__nav}>
				<Link to='/' className={`${style.header__nav__block} ${location.pathname === '/' ? style.active : ''}`}>
					Генерация
				</Link>
				<Link to='/prev-generations' className={`${style.header__nav__block} ${location.pathname === '/prev-generations' ? style.active : ''}`}>
					История
				</Link>
			</nav>
			<p className={style.header__name}>Winstrike</p>
		</div>
	)
}
