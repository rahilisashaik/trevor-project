import Image from "next/image";

export default function Home() {
  return (
    <div className="grid grid-rows-[20px_1fr_20px] items-center justify-items-center min-h-screen p-8 pb-20 gap-16 sm:p-20 font-[family-name:var(--font-geist-sans)]">
      
      <header>Kevin Guo</header>
      <main className="flex flex-col gap-8 items-center justify-center min-h-screen">

        <p>
          <img id="headshot" src="kevin.jpeg" alt="Kevin Guo"/>
        </p>

        <p id="facts">
          <b>Education:</b> The University of California, Berkeley <br /> <br />
          <b>Major:</b> Electrical Engineering & Computer Science + Business Administration
        </p>
      
        <p>
          <b>Biography: </b>Hello! My name is Kevin Guo. 
          I am currently a student studying Electrical Engineering and Computer Science + Business Administration in the Management, Entrepreneurship, 
          and Technology (M.E.T) program at UC Berkeley. I am fascinated by the unique relationships/innovations made possible through the intersection of business and engineering. Outside
          of my academic/professional goals, I enjoy a variety of activities, including weight lifting, tennis, and playing the flute!
        </p>
      </main>
    </div>
  );
}
