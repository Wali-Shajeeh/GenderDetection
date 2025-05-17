const App = () => {
         const videoRef = React.useRef(null);
         const canvasRef = React.useRef(null);
         const [status, setStatus] = React.useState('Waiting for camera access...');
         const [result, setResult] = React.useState('');
         const [isCapturing, setIsCapturing] = React.useState(false);
         const socketRef = React.useRef(null);
         const timeoutRef = React.useRef(null);

         React.useEffect(() => {
             // Block reloads
             window.location.reload = () => console.log('Reload blocked');
             window.onbeforeunload = () => null;
             window.addEventListener('error', (e) => {
                 console.error('Global error:', e.message, e);
                 return true;
             });
             window.addEventListener('unhandledrejection', (e) => {
                 console.error('Unhandled promise rejection:', e.reason);
                 e.preventDefault();
             });
             window.onerror = (msg, url, line, col, error) => {
                 console.error(`Error: ${msg} at ${url}:${line}:${col}`, error);
                 return true;
             };

             // Initialize SocketIO
             socketRef.current = io('https://5000-01jvevdd68qh21f5w9789g8977.cloudspaces.litng.ai, {
                 transports: ['websocket'],
                 reconnection: true,
                 reconnectionAttempts: Infinity,
                 reconnectionDelay: 1000
             });

             socketRef.current.on('connect', () => {
                 console.log('SocketIO connected');
                 setStatus('Connected to server');
             });

             socketRef.current.on('status', (data) => {
                 console.log('Status:', data.message);
                 setStatus(data.message);
             });

             socketRef.current.on('result', (data) => {
                 try {
                     clearTimeout(timeoutRef.current);
                     setResult(data.message);
                     setStatus('Processing complete');
                     setIsCapturing(false);
                     console.log('Result displayed:', data.message);
                 } catch (e) {
                     console.error('Error displaying result:', e);
                     setStatus('Error displaying result');
                     setIsCapturing(false);
                 }
             });

             socketRef.current.on('disconnect', () => {
                 console.log('SocketIO disconnected');
                 setStatus('Disconnected from server. Reconnecting...');
             });

             socketRef.current.on('connect_error', (error) => {
                 console.error('SocketIO error:', error.message, error);
                 setStatus('Connection error. Check console.');
             });

             // Start camera
             const startCamera = async () => {
                 try {
                     const stream = await navigator.mediaDevices.getUserMedia({
                         video: { width: 640, height: 480 }
                     });
                     if (videoRef.current) {
                         videoRef.current.srcObject = stream;
                         setStatus('Camera connected');
                         console.log('Camera stream initialized');
                     }
                 } catch (err) {
                     console.error('Camera access error:', err);
                     setStatus(`Failed to access camera: ${err.message}`);
                 }
             };
             startCamera();

             // Cleanup
             return () => {
                 socketRef.current.disconnect();
                 if (videoRef.current && videoRef.current.srcObject) {
                     videoRef.current.srcObject.getTracks().forEach(track => track.stop());
                 }
             };
         }, []);

         const captureImage = () => {
             if (!videoRef.current || videoRef.current.readyState !== videoRef.current.HAVE_ENOUGH_DATA) {
                 console.error('Video not ready');
                 setStatus('Camera not ready');
                 return;
             }
             try {
                 const canvas = canvasRef.current;
                 canvas.width = videoRef.current.videoWidth;
                 canvas.height = videoRef.current.videoHeight;
                 const ctx = canvas.getContext('2d');
                 ctx.drawImage(videoRef.current, 0, 0, canvas.width, canvas.height);
                 const imageData = canvas.toDataURL('image/jpeg', 0.8);
                 const base64Data = imageData.split(',')[1];
                 socketRef.current.emit('image', base64Data);
                 setStatus('Capturing image...');
                 setResult('');
                 setIsCapturing(true);
                 console.log('Image captured and sent');
                 timeoutRef.current = setTimeout(() => {
                     setStatus('Processing failed. Try again.');
                     setResult('');
                     setIsCapturing(false);
                 }, 10000);
             } catch (e) {
                 console.error('Error capturing image:', e);
                 setStatus('Error capturing image');
                 setIsCapturing(false);
             }
         };

         return (
             <div className="flex flex-col items-center justify-center min-h-screen bg-gray-100">
                 <h1 className="text-3xl font-bold mb-4">Gender Classification</h1>
                 <div className="bg-white p-6 rounded-lg shadow-lg">
                     <video
                         ref={videoRef}
                         autoPlay
                         className="w-full max-w-md rounded mb-4"
                     ></video>
                     <canvas ref={canvasRef} className="hidden"></canvas>
                     <button
                         onClick={captureImage}
                         disabled={isCapturing}
                         className={`px-4 py-2 rounded text-white ${
                             isCapturing ? 'bg-blue-300 cursor-not-allowed' : 'bg-blue-500 hover:bg-blue-600'
                         } mb-4`}
                     >
                         Capture
                     </button>
                     <p className="text-center text-gray-800 font-semibold mt-2">{result}</p>
                     <p className="text-center text-gray-600 mt-2">{status}</p>
                 </div>
             </div>
         );
     };

     window.App = App;
