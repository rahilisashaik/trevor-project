import Image from "next/image";

export default function Home() {
  return (
    <div className="grid grid-rows-[20px_1fr_20px] items-center justify-items-center min-h-screen p-8 pb-20 gap-16 sm:p-20 font-[family-name:var(--font-geist-sans)]">
      <main className="flex flex-col gap-8 row-start-2 items-center sm:items-start">
        
        <h1>Hi! I'm Emily Mo.</h1>
          
        
        <div className="text">
          <p>Right now, I'm studying EECS and Business under UC Berkeley's <a href="https://www.businessinsider.com/uc-berkeley-met-for-future-tech-leaders-2017-4"><u><b>M.E.T. Program</b></u></a>. I originally come from a background in mostly finance and business, but I am currently exploring my passion in technology and machine learning. Here's what I'm up to: </p>
        </div>  

        <div className="list">
          <ul>
              <li>Barclays | Incoming Technology Analyst Intern | Jun 2025</li>
              <li>The Trevor Project | Technical Consultant | Feb 2025</li>
              <li>Quest Diagnostics | Machine Learning Engineer | Aug 2024 - Jan 2025</li>
              <li>Susquehanna International Group | Women's Trading Program | Nov 2024 </li>
              <li>University of Washington | Research Intern | Jan 2024 - Dec 2024</li>
              <li>AI Data Innovations | Project Intern | Jun 2023 - Aug 2023</li>

          </ul>
        </div>  

        <div className="text">
          <p>For fun, I enjoy training with Cal's Kung Fu Team, updating my <a href="https://www.strava.com/athletes/123047488"><u><b>Strava</b></u></a>, and listening to jazz.</p>
          <p>Don't be a stranger, reach out:    
            <a href="mailto:emilymo999@berkeley.edu" className="no-underline"><i className="fa-solid fa-envelope"></i></a>     
            <a href="https://www.linkedin.com/in/emiilymo" className="no-underline"><i className="fa-brands fa-linkedin"></i></a>    
            <a href="https://github.com/emilymo999" className="no-underline"><i className="fa-brands fa-github"></i></a>

          </p>
        </div>  

      </main>
      <footer className="row-start-3 flex gap-6 flex-wrap items-center justify-center">
      </footer>
    </div>
  );
}
