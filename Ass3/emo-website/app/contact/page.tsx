export default function Contact() {
    return (
      <div className="min-h-screen flex flex-col items-center justify-center p-8">
        
        <h1 className="text-3xl font-bold mb-4 text-center">Contact Me 🔥🔥🔥</h1>
  
        {/* Contact Form */}
        <form className="w-full max-w-lg bg-white shadow-md rounded-lg p-6 space-y-4">
          <div>
            <label className="block text-gray-700 font-medium mb-2">Name</label>
            <input
              type="text"
              className="w-full px-4 py-2 border rounded-lg focus:ring-2 focus:ring-blue-500"
              placeholder="Your Name"
              required
            />
          </div>
  
          <div>
            <label className="block text-gray-700 font-medium mb-2">Email</label>
            <input
              type="email"
              className="w-full px-4 py-2 border rounded-lg focus:ring-2 focus:ring-blue-500"
              placeholder="your@email.com"
              required
            />
          </div>
  
          <div>
            <label className="block text-gray-700 font-medium mb-2">Message</label>
            <textarea
              className="w-full px-4 py-2 border rounded-lg focus:ring-2 focus:ring-blue-500"
              rows={4}
              placeholder="Your message..."
              required
            ></textarea>
          </div>
  
          <button
            type="submit"
            className="w-full bg-blue-600 text-white py-2 rounded-lg hover:bg-blue-700 transition"
          >
            Send Message
          </button>
        </form>
  
        {/* Contact Links */}
        <div className="mt-8 text-center opacity-0 animate-fade-in delay-500">
          <p className="text-gray-700">Or reach me directly:</p>
          <p className="text-blue-500">
            📧 <a href="mailto:your@email.com">your@email.com</a>
          </p>
          <p className="text-blue-500">
            🔗 <a href="https://linkedin.com/in/yourprofile" target="_blank" rel="noopener noreferrer">
              LinkedIn
            </a>
          </p>
        </div>
      </div>
    );
  }
  