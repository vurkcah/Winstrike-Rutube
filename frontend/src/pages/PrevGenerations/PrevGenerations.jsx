import React, { useState, useEffect } from 'react'
import axios from 'axios'
import './PrevGenerations.scss'

const Modal = ({ show, onClose, content }) => {
	if (!show || !content) return null

	return (
		<div className='modal-overlay' onClick={onClose}>
			<div className='modal' onClick={(e) => e.stopPropagation()}>
				<button className='modal-close' onClick={onClose}>
					×
				</button>
				<div className='modal-content'>
					<h2>Название: {content.title}</h2>
					{content.frame_image ? <img src={`data:image/jpeg;base64,${content.frame_image}`} alt={content.title} className='modal-image' /> : <div className='modal-no-image'>Изображение недоступно</div>}
					<p>
						Статус: <span className={content.status === 'успешно' ? 'status-loaded' : 'status-error'}>{content.status}</span>
					</p>
					<p className='modal-description'>Описание: {content.description}</p>
					<div className='modal-tags'>
						<h3>Категории:</h3>
						<div className='modal-tags__cards'>
							{content.categories && content.categories.length > 0 ? (
								content.categories.map((category, index) => (
									<span key={index} className='modal-tag'>
										#{category}
									</span>
								))
							) : (
								<p>Категории отсутствуют</p>
							)}
						</div>
					</div>
				</div>
			</div>
		</div>
	)
}

export const PrevGenerations = () => {
	const [modalContent, setModalContent] = useState(null)
	const [showModal, setShowModal] = useState(false)
	const [cardsData, setCardsData] = useState([])
	const [loading, setLoading] = useState(true)

	useEffect(() => {
		const fetchVideos = async () => {
			try {
				const response = await axios.get('http://localhost:5000/videos')
				if (response.status === 200) {
					setCardsData(response.data.videos)
				}
			} catch (error) {
				console.error('Ошибка при получении данных:', error)
			} finally {
				setLoading(false)
			}
		}

		fetchVideos()
	}, [])

	const openModal = (content) => {
		setModalContent(content)
		setShowModal(true)
	}

	const closeModal = () => {
		setShowModal(false)
		setModalContent(null)
	}

	return (
		<div className='container'>
			<div className='main-box'>
				<div className='downloads'>
					<h1 className='downloads-title'>Предыдущие загрузки</h1>
					<div className='prev-gens'>
						{loading ? (
							<p className='prev-gens--loads'>Загрузка данных...</p>
						) : cardsData.length === 0 ? (
							<p className='prev-gens--dontloads'>Нет загруженных видео.</p>
						) : (
							cardsData.map((card) => (
								<div key={card.video_id} className='prev-gens__card'>
									<div className='prev-gens__card__content'>
										<p className='prev-gens__card__top-name'>
											Название: <span>{card.title}</span>
										</p>
										<p className='prev-gens__card__top-status'>
											Статус: <span className={card.status === 'успешно' ? 'status-loaded' : 'status-error'}>{card.status}</span>
										</p>
									</div>
									{card.status !== 'загрузка' && (
										<button className='prev-gens__card__btn' onClick={() => openModal(card)}>
											Перейти
										</button>
									)}
								</div>
							))
						)}
					</div>
				</div>
			</div>
			<Modal show={showModal} onClose={closeModal} content={modalContent} />
		</div>
	)
}

export default PrevGenerations
