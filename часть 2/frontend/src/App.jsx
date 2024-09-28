import { Header } from './components/Header/Header'

import { Routes, Route } from 'react-router-dom'
import {Generation} from "./pages/Generation/Generation.jsx";
import {PrevGenerations} from "./pages/PrevGenerations/PrevGenerations.jsx";
import {NotFound} from "./pages/NotFound/NotFound.jsx";

function App() {
	return (
		<>
			<div className='main'>
				<Header />
				<Routes>
					<Route path="/" element={<Generation/>} />
					<Route path="/prev-generations" element={<PrevGenerations />} />
					<Route path="*" element={<NotFound />} />
				</Routes>
			</div>
		</>
	)
}

export default App
