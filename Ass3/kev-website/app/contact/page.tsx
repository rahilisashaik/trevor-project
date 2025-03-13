"use client";

import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs"
import {
    Card,
    CardContent,
    CardDescription,
    CardFooter,
    CardHeader,
    CardTitle,
  } from "@/components/ui/card"
  
  import {
    HoverCard,
    HoverCardContent,
    HoverCardTrigger,
  } from "@/components/ui/hover-card"

  import { Textarea } from "@/components/ui/textarea"

  

export default function ContactPage() {
    return (
      <main className="flex flex-col items-center justify-center min-h-screen p-8">
        <h1 className="text-3xl font-bold">Contact Me</h1>
        <Tabs defaultValue="account" className="flex flex-col items-center justify-center min-h-screen w-[400px]">
            <TabsList>
                <TabsTrigger value="account">Contact Card</TabsTrigger>
                <TabsTrigger value="password">Socials</TabsTrigger>
            </TabsList>
            <TabsContent value="account">
                <Card className = "w-[400px]">
                    <CardHeader>
                        <CardTitle className = "text-[20px] font-semibold">Kevin Guo</CardTitle>
                        <CardDescription>Reach out to employ me (please)</CardDescription>
                    </CardHeader>
                    <CardContent>
                        <p className = "text-lg">
                            <HoverCard>
                                <HoverCardTrigger className = "text-[15px] font-semibold">
                                    Email: kguo28@berkeley.edu
                                </HoverCardTrigger>
                                <HoverCardContent className = "text-[15px] font-semibold">
                                    Personal: kevinmuguo@gmail.com
                                </HoverCardContent>

                                <br />

                                <HoverCardTrigger className = "text-[15px] font-semibold">
                                    Phone: (248) 805-2523
                                </HoverCardTrigger>
                                <HoverCardContent className = "text-[15px] font-semibold">
                                    Personal: kevinmuguo@gmail.com
                                </HoverCardContent>

                            </HoverCard>
                        </p>
                        <p className = "text-[15px] font-semibold">
                            Leave me a message:
                            <Textarea />
                        </p>
                    </CardContent>
                </Card>
            </TabsContent>
            <TabsContent value="password">

            <Card className = "w-[400px]">
                    <CardHeader>
                        <CardTitle className = "text-[20px] font-semibold">Kevin's Socials</CardTitle>
                        <CardDescription>Don't blow up my dm's!</CardDescription>
                    </CardHeader>
                    <CardContent>
                        <div className = "flex gap-4 w-[200px]" id = "contacts">
                            <div className="relative group w-12 h-12">
                                <img
                                src="/instagram (1).png"
                                alt="Instagram"
                                className="absolute w-full h-full transition-opacity duration-500 opacity-100 group-hover:opacity-0"
                                />
                                <img
                                src="/instagram (2).png"
                                alt="Instagram Hover"
                                className="absolute w-full h-full transition-opacity duration-500 opacity-0 group-hover:opacity-100"
                                />
                            </div>

                            <div className="relative group w-12 h-12">
                                <img
                                src="/linkedin.png"
                                alt="LinkedIn"
                                className="absolute w-full h-full transition-opacity duration-500 opacity-100 group-hover:opacity-0"
                                />
                                <img
                                src="/linkedin (1).png"
                                alt="LinkedIn Hover"
                                className="absolute w-full h-full transition-opacity duration-500 opacity-0 group-hover:opacity-100"
                                />
                            </div>

                        </div>
                    </CardContent>
                </Card>
            </TabsContent>
        </Tabs>

        <p>Email: kguo28@berkeley.edu</p>
        <p>Phone: (248) 805-2523</p>
      </main>
    );
  }