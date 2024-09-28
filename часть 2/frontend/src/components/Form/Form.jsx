import React, { useState } from 'react'
import axios from 'axios'
import { Toast, ToastDescription, ToastProvider, ToastTitle, ToastViewport } from '@radix-ui/react-toast'
import { Link } from 'react-router-dom'
import style from './Form.module.scss'

const Form = () => {
	const [open, setOpen] = useState(false)
	const [videoTitle, setVideoTitle] = useState('')
	const [videoDescription, setVideoDescription] = useState('')
	const [videoFile, setVideoFile] = useState(null)
	const [isLoading, setIsLoading] = useState(false) // Состояние для загрузки

	const handleSubmit = async (e) => {
		e.preventDefault()

		if (!videoFile) {
			alert('Пожалуйста, выберите видеофайл')
			return
		}

		const formData = new FormData()
		formData.append('video', videoFile)
		formData.append('title', videoTitle)
		formData.append('description', videoDescription)

		setIsLoading(true) // Показать лоадер

		try {
			// Отправка данных на бэкенд
			const response = await axios.post('http://localhost:5000/upload', formData, {
				headers: {
					'Content-Type': 'multipart/form-data',
				},
			})

			if (response.status === 200) {
				// Обработка успешного ответа
				console.log('Ответ от сервера:', response.data)
				setOpen(true) // Покажем уведомление после успешной отправки
				setTimeout(() => setOpen(false), 6000) // Уведомление исчезает через 6 секунд
			}
		} catch (error) {
			console.error('Ошибка при отправке данных:', error)
			alert('Произошла ошибка при отправке данных на сервер')
		} finally {
			setIsLoading(false) // Скрыть лоадер после завершения запроса
		}
	}

	const handleTitleChange = (e) => {
		setVideoTitle(e.target.value)
	}

	const handleDescriptionChange = (e) => {
		setVideoDescription(e.target.value)
	}

	const handleFileChange = (e) => {
		setVideoFile(e.target.files[0])
	}

	return (
		<div className='--form-container'>
			<ToastProvider swipeDirection='right'>
				<form className={style.form} onSubmit={handleSubmit}>
					{/* Поле для загрузки видео */}
					<label htmlFor='video' className={style.form__drop} id='dropcontainer'>
						<span className={style.titled}>Нажмите на это окно</span> или
						<input type='file' id='video' accept='video/*' required onChange={handleFileChange} />
					</label>

					{/* Поле для ввода названия видео */}
					<label htmlFor='title' className={style.form__label}>
						<span className={style.titled}>Название видео</span>
						<input type='text' id='title' placeholder='Введите название' required onChange={handleTitleChange} />
					</label>

					{/* Поле для ввода описания видео */}
					<label htmlFor='description' className={style.form__label}>
						<span className={style.titled}>Описание видео</span>
						<textarea id='description' placeholder='Введите описание' required onChange={handleDescriptionChange} />
					</label>

					{/* Кнопка отправки */}
					<button type='submit' className={style.form__submit}>
						Загрузить видео
					</button>
				</form>

				{/* Toast уведомление */}
				<Toast open={open} onOpenChange={setOpen} className={style.toastRoot}>
					<ToastTitle>
						<p>
							Запрос под названием <span>"{videoTitle || 'название видео не указано'}"</span> создан
						</p>
					</ToastTitle>
					<ToastDescription asChild>
						<Link to='/prev-generations' className={style.link}>
							Перейти к предыдущим генерациям
						</Link>
					</ToastDescription>
				</Toast>

				{/* Контейнер для отображения уведомлений */}
				<ToastViewport className={style.toastViewport} />
			</ToastProvider>

			{/* Загрузочный индикатор */}
			{isLoading && (
				<div className='loaderOverlay'>
					<div className='loader'></div>
				</div>
			)}
		</div>
	)
}

export default Form
