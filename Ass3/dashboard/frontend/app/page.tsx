"use client";
import {
  Card,
  CardContent,
  CardDescription,
  CardFooter,
  CardHeader,
  CardTitle,
} from "@/components/ui/card"

export default function Home() {
  return (
    <div>
      <main>
      <Card>
        <CardHeader>
          <CardTitle><b> PoopyButt </ b></CardTitle>
          <CardDescription>Patient Info</CardDescription>
        </CardHeader>
        <CardContent>
          <p><b>Phone Number: </ b> 696-696-6969</p>
        </CardContent>
        <CardContent>
          <p><b>Time of Last Call: </ b> 07: 2025-07-15</p>
        </CardContent>
        <CardContent>
          <p>Card Content</p>
        </CardContent>
        <CardFooter>
          <p>Card Footer</p>
        </CardFooter>
      </Card>

      </main>
      <footer>
      </footer>
    </div>
  );
} 

